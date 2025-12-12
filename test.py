import sys, os, io, time, logging
import numpy as np
import torch
import torch.multiprocessing
import pytorch_lightning as pl
import matplotlib
import hydra
import wandb
import time

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from omegaconf import DictConfig, OmegaConf

from datasets.semseg_datamodule import SegmentationDataModule
from models.semseg_plm import SegmentationModel,ProtoSegModel
from utils.utils import make_dir, shrink_dict, getListOfFiles, natural_keys
from utils.eval_utils import evaluate_model, evaluate_model_samplewise, visualize_results, visualize_save_confusions, visualize_scores_per_class, visualize_confusion_genus, count_errors_by_distance
from utils.transform_utils import get_transforms

""" 
This script is used to test a pre-trained segmentation model using a Hydra configuration.
It supports:

- Loading a trained checkpoint and setting up the model.
- Preparing the data using SegmentationDataModule with test transforms.
- Evaluating the model both dataset-wise and sample-wise.
- Computing quantitative metrics (F1, IoU, Precision, Recall) and error analysis by distance.
- Visualizing results, confusion matrices, and qualitative predictions.
- Logging metrics to Weights & Biases (WandB).

Note: Adjust the config files and checkpoint paths before running.

"""

# # this can avoid avoid shared memory allocation errors
torch.multiprocessing.set_sharing_strategy('file_system')

# # set the float32 matrix multiplication precision to 'medium'
torch.set_float32_matmul_precision('medium')


#%% Hydra
@hydra.main(version_base=None, config_path="configs", config_name="test")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # # get device, set matplotlib style
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    matplotlib.rcParams.update(cfg.matplotlib_style)
    matplotlib.use(cfg.matplotlib_backend)

    # # set pytorch lightning seed
    if cfg.pl_seed:
        seed_everything(cfg.pl_seed, workers=True)
    
    #%% dataModule
    test_transforms = get_transforms(cfg.data.test_transforms)
    dataModule = SegmentationDataModule(cfg.data.image_dir, cfg.model.num_classes, test_transforms, test_transforms, test_transforms, value_mapping=cfg.data.value_mapping, ignore_index=cfg.data.ignore_index, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)

    # # setup dataModule
    dataModule.prepare_data()
    dataModule.setup()
    print('# Train imgs: ', len(dataModule.train_dataset))
    
    # # visualize dims of single image + mask
    for idx, batch in enumerate(dataModule.train_dataloader()):
        try:
            images = batch['image']
            masks = batch['mask']
            print('Image batch shape: ', images.shape, '\nMask batch shape:  ', masks.shape)
            break
        except Exception as e:
            print(f"Error at index {idx}: {e}")
    
    #%% testing
    # # load model with ckpt_type checkpoint
    ckpts_paths = getListOfFiles(cfg.ckpt_path)
    ckpts_paths.sort(key=natural_keys)
    ckpt_path = [s for s in ckpts_paths if cfg.ckpt_type in s][-1]
    ckpt_name = ckpt_path[ckpt_path.rindex('/')+1:-5]
    print('ckpt_path: ', ckpt_path)
    print('ckpt_name: ', ckpt_name)
    cfg.ckpt_type = ckpt_name 
    OmegaConf.resolve(cfg) 
    model = SegmentationModel.load_from_checkpoint(ckpt_path).to(device)
    
    #change evaluation matrix:
    D_path_eval = cfg.d_matrix_eval 
    d_matrix_eval= torch.load(D_path_eval)
    model.hparams.d_matrix_eval=d_matrix_eval
    model.test_AHC.D = d_matrix_eval

    #add information about genus to index and specie to index
    genus_to_index=cfg.genus.genus_to_index
    specie_to_genus=cfg.genus.specie_to_genus_index
    model.hparams.genus_to_index=genus_to_index
    model.hparams.specie_to_genus=specie_to_genus

    #set pixel as background if all predictions are below a certain treshold
    #not recomended (didnt work that good)
    threshold = None
    print(f"!Threshold set to {threshold}")
    name = f"{cfg.exp_name}_test_{threshold}"

    # initialize logger, run test loop on the model 
    logger = WandbLogger(name=name, project=cfg.log_name)
    trainer = pl.Trainer(logger=logger)
    trainer.test(model=model, dataloaders=dataModule)   
    
    # # all class names and colors in order of index
    class_names = [cfg.data.classes[i].name for i in sorted(cfg.data.classes.keys(), key=int)]
    class_colors = [cfg.data.classes[i].color for i in sorted(cfg.data.classes.keys(), key=int)]
    labels_sorted = [k for k,_ in sorted(model.hparams.genus_to_index.items(), key=lambda x: x[1])]
    
    #%% dataset-wise evaluation
    test_loader = dataModule.test_dataloader()
    metrics_dir = make_dir(cfg.test_path,'quantitative')
    
    # # # Classic Top-1   
    metrics_dir_top_1 = make_dir(metrics_dir, 'top_1')
    metrics = evaluate_model(model, test_loader, cfg.model.num_classes, device, custom_avg=cfg.data.class_groups, save_dir=metrics_dir_top_1,threshold=threshold)
    visualize_scores_per_class(shrink_dict(metrics['semseg'], ['F1','IoU','Precision','Recall']), os.path.join(metrics_dir_top_1, 'semseg_scores.png'), class_names)
    visualize_save_confusions(metrics['conf'],metrics_dir_top_1,display_labels=class_names)
    visualize_confusion_genus(metrics['conf_genus'], save_dir= metrics_dir_top_1, display_labels=labels_sorted)
    print(f"Avg class-wise F1 scores - Top1Acc:  {np.array2string(metrics['semseg']['F1'], formatter={'float_kind': lambda x: f'{x:.3f}'})}")
    
    # count errors by distance
    errors_by_dist ,tot_errors = count_errors_by_distance(metrics['conf'], distance_matrix=d_matrix_eval,bins_edges=[0.0, 3.0, 7.0, 10.1])
    print("Errors by distance:")
    for dist, count in errors_by_dist.items():
        print("Interval: ", dist, "relative_errors", count/tot_errors)

    # get metrics per class to WandB
    columns = ["model_name", "class_name", "F1", "IoU", "Precision", "Recall"]
    metrics_table = wandb.Table(columns=columns)

    #log metrics per class
    for idx, class_name in enumerate(class_names):
        metrics_table.add_data(
            cfg.exp_name,                    # model/run name
            class_name,
            metrics['semseg']['F1'][idx],
            metrics['semseg']['IoU'][idx],
            metrics['semseg']['Precision'][idx],
            metrics['semseg']['Recall'][idx]
        )
    wandb.log({"per_class_metrics": metrics_table})
    wandb.log({"test_AHC_D_name": D_path_eval})


    #%% sample-wise evaluation and qualitative visualization (pick-data)
    pick_loader = dataModule.pick_dataloader()
    qual_dir = make_dir(cfg.test_path,'qualitative')


    # # Classic Top-1
    qual_dir_top_1 = make_dir(qual_dir, 'top_1')
    all_imgs, all_metrics, flt_metrics = evaluate_model_samplewise(model, pick_loader, cfg.model.num_classes, device, save_metrics_batch_limit=10, save_imgs_batch_limit=10)
    visualize_results(all_imgs['imgs'], all_imgs['masks'], all_imgs['preds'], colors=class_colors, variances=all_imgs['var'], num_samples=5, which='first', save_dir=qual_dir_top_1, class_names=class_names)
    visualize_results(all_imgs['imgs'], all_imgs['masks'], all_imgs['preds'], colors=class_colors, variances=all_imgs['var'], num_samples=5, which='last', save_dir=qual_dir_top_1, class_names=class_names)
    visualize_results(all_imgs['imgs'], all_imgs['masks'], all_imgs['preds'], colors=class_colors, variances=all_imgs['var'], num_samples=4, which='random', save_dir=qual_dir_top_1, class_names=class_names)

    #%% re initialize hydra
    hydra.core.global_hydra.GlobalHydra.instance().clear()

#%%    
if __name__ == "__main__":
    main()