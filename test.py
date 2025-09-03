import sys, os, io, time, logging
import numpy as np
import torch
import torch.multiprocessing
import pytorch_lightning as pl
import matplotlib
import hydra

from pytorch_lightning import seed_everything
from omegaconf import DictConfig, OmegaConf

from datasets.semseg_datamodule import SegmentationDataModule
from models.semseg_plm import SegmentationModel
from utils.utils import make_dir, shrink_dict, getListOfFiles, natural_keys
from utils.eval_utils import evaluate_model, evaluate_model_samplewise, visualize_results, visualize_save_confusions, visualize_scores_per_class
from utils.transform_utils import get_transforms

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
    cfg.ckpt_type = ckpt_name # adjust ckpt_type to get also the epoch number in the test file path (e.g. not only "best" but "best_epoch=451.ckpt")
    OmegaConf.resolve(cfg) # since test_path is dynamically set in the hydra config we need to run the interpolation again with modified ckpt_type
    # # load model with ckpt_path weights
    model = SegmentationModel.load_from_checkpoint(ckpt_path).to(device)
    # # set_patch_2_img_size
    # model.set_patch_2_img_size(False)
    # # run test_step
    trainer = pl.Trainer(logger=False)
    trainer.test(model=model, dataloaders=dataModule)
    
    # # all class names and colors in order of index
    class_names = [cfg.data.classes[i].name for i in sorted(cfg.data.classes.keys(), key=int)]
    class_colors = [cfg.data.classes[i].color for i in sorted(cfg.data.classes.keys(), key=int)]
    
    
    #%% dataset-wise evaluation
    test_loader = dataModule.test_dataloader()
    metrics_dir = make_dir(cfg.test_path,'quantitative')
    
    # # # Classic Top-1
    metrics_dir_top_1 = make_dir(metrics_dir, 'top_1')
    metrics = evaluate_model(model, test_loader, cfg.model.num_classes, device, custom_avg=cfg.data.class_groups, save_dir=metrics_dir_top_1)
    visualize_scores_per_class(shrink_dict(metrics['semseg'], ['F1','IoU','Precision','Recall']), os.path.join(metrics_dir_top_1, 'semseg_scores.png'), class_names)
    visualize_save_confusions(metrics['conf'],metrics_dir_top_1,display_labels=class_names)
    print(f"Avg class-wise F1 scores - Top1Acc:  {np.array2string(metrics['semseg']['F1'], formatter={'float_kind': lambda x: f'{x:.3f}'})}")
    
    
    #%% sample-wise evaluation and qualitative visualization (pick-data)
    pick_loader = dataModule.pick_dataloader()
    qual_dir = make_dir(cfg.test_path,'qualitative')

    # # Classic Top-1
    qual_dir_top_1 = make_dir(qual_dir, 'top_1')
    all_imgs, all_metrics, flt_metrics = evaluate_model_samplewise(model, pick_loader, cfg.model.num_classes, device, save_metrics_batch_limit=10, save_imgs_batch_limit=10)
    visualize_results(all_imgs['imgs'], all_imgs['masks'], all_imgs['preds'], colors=class_colors, variances=all_imgs['var'], num_samples=5, which='first', save_dir=qual_dir_top_1, class_names=class_names)
    visualize_results(all_imgs['imgs'], all_imgs['masks'], all_imgs['preds'], colors=class_colors, variances=all_imgs['var'], num_samples=5, which='last', save_dir=qual_dir_top_1, class_names=class_names)


    #%% re initialize hydra
    hydra.core.global_hydra.GlobalHydra.instance().clear()


#%%    
if __name__ == "__main__":
    main()