import sys, os, io, time, logging
import numpy as np
import torch
import torch.multiprocessing
import pytorch_lightning as pl
import matplotlib
import wandb
import hydra

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from omegaconf import DictConfig, OmegaConf
from collections import Counter

from datasets.semseg_datamodule import SegmentationDataModule
from models.semseg_plm import SegmentationModel
from utils.utils import make_dir, shrink_dict
from utils.eval_utils import evaluate_model, evaluate_model_samplewise, visualize_results, visualize_save_confusions, visualize_scores_per_class
from utils.transform_utils import get_transforms
from utils.callback_utils import get_callbacks

# # this can avoid avoid shared memory allocation errors
torch.multiprocessing.set_sharing_strategy('file_system')

# # set the float32 matrix multiplication precision to 'medium'
torch.set_float32_matmul_precision('medium')


#%% Hydra
@hydra.main(version_base=None, config_path="configs", config_name="train")
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
    train_transforms = get_transforms(cfg.data.train_transforms)
    val_transforms = get_transforms(cfg.data.val_transforms)
    test_transforms = get_transforms(cfg.data.test_transforms)
    dataModule = SegmentationDataModule(cfg.data.image_dir, cfg.model.num_classes, train_transforms, val_transforms, test_transforms, weightedSampling=cfg.data.weightedSampling, value_mapping=cfg.data.value_mapping, ignore_index=cfg.data.ignore_index, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)

    # # setup dataModule
    dataModule.prepare_data()
    dataModule.setup()
    print('# Train imgs: ', len(dataModule.train_dataset))
    
    # # get statistics of dataset
    # if cfg.data.weightedSampling:     
    #     class_presence = dataModule.train_dataset.class_presence
    #     total = Counter()
    #     for d in class_presence:
    #         total.update(d)
    #     total_class_presence = dict(total)
    #     get_keys = [key for key, value in total_class_presence.items() if value >1 and value <=50]
    #     print(get_keys)
    
    # # visualize dims of single image + mask
    for idx, batch in enumerate(dataModule.train_dataloader()):
        try:
            images = batch['image']
            masks = batch['mask']
            print('Image batch shape: ', images.shape, '\nMask batch shape:  ', masks.shape)
            break
        except Exception as e:
            print(f"Error at index {idx}: {e}")
    
    
    
    #%% compute class-balanced weights (effective numbers) if needed
    cb_alpha = None
    if cfg.model.loss == 'CBFocal':
        # Derive per-class sample counts (image-level presence) from train dataset
        train_ds = dataModule.train_dataset
        if hasattr(train_ds, 'class_presence'):
            class_presence = train_ds.class_presence
        else:
            class_presence = train_ds.calculate_class_presence()
        # Sum presence across dataset to get counts per class
        counts = np.zeros(cfg.model.num_classes, dtype=np.int64)
        for presence in class_presence:
            for cls_id, present in presence.items():
                counts[int(cls_id)] += int(present)
        # Compute effective alphas: (1 - beta) / (1 - beta^n_i)
        beta = float(cfg.model.cb_beta)
        counts = counts.astype(np.float64)
        # Avoid beta**0 division by zero; where counts == 0, set alpha to 0
        with np.errstate(divide='ignore', invalid='ignore'):
            effective_nums = (1.0 - np.power(beta, counts)) / (1.0 - beta)
        alphas = np.zeros_like(effective_nums, dtype=np.float64)
        nonzero = effective_nums > 0
        alphas[nonzero] = (1.0 / effective_nums[nonzero])
        # Optional normalization
        if bool(cfg.model.cb_alpha_normalize):
            s = alphas.sum()
            if s > 0:
                alphas = alphas / s
        cb_alpha = torch.tensor(alphas, dtype=torch.float32)

    # Decide ignore_index for loss/metrics (optionally ignore background)
    ignore_index_for_loss = 0 if getattr(cfg.model, 'ignore_background', False) else cfg.data.ignore_index

    #%% model
    model = SegmentationModel(
        cfg.model.model,
        cfg.model.encoder_name,
        cfg.model.img_size,
        cfg.model.num_classes,
        cfg.model.lr,
        ignore_index=ignore_index_for_loss,
        optimizer=cfg.model.optimizer,
        lr_scheduler=cfg.model.lr_scheduler,
        loss=cfg.model.loss,
        weight=cfg.model.weight,
        patch_2_img_size=cfg.model.patch_2_img_size,
        focal_gamma=cfg.model.focal_gamma,
        cb_alpha=cb_alpha,
    )
    
    
    #%% training
    callbacks = get_callbacks(cfg.model.callbacks)
    logger = WandbLogger(name=cfg.exp_name, project=cfg.log_name, entity=cfg.entity)
    # Log CB alpha stats and distribution plot if available
    if cb_alpha is not None:
        try:
            import matplotlib.pyplot as plt
            # Scalars
            alpha_np = cb_alpha.detach().cpu().numpy()
            max_w = float(alpha_np.max()) if alpha_np.size else 0.0
            # min over non-zeros to avoid trivial zeros from empty classes
            nz = alpha_np[alpha_np > 0]
            min_w = float(nz.min()) if nz.size else 0.0
            logger.experiment.log({
                'cb_alpha_max': max_w,
                'cb_alpha_min_nonzero': min_w,
            })
            # Plot distribution
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.bar(np.arange(len(alpha_np)), alpha_np)
            ax.set_title('CB Alpha Distribution')
            ax.set_xlabel('Class index')
            ax.set_ylabel('Alpha')
            ax.grid(True, axis='y', linestyle='--', alpha=0.3)
            logger.experiment.log({'cb_alpha_plot': wandb.Image(fig)})
            plt.close(fig)
        except Exception as e:
            print(f"Warning: failed to log CB alpha stats: {e}")
    trainer = pl.Trainer(max_epochs=cfg.model.max_epochs, callbacks=callbacks, logger=logger, deterministic=cfg.deterministic_train)
    last_ckpt_path = os.path.join(cfg.log_path,'checkpoints/last.ckpt')
    ckpt_path_resume = last_ckpt_path if os.path.exists(last_ckpt_path) else None
    start_time = time.time()
    trainer.fit(model, dataModule, ckpt_path=ckpt_path_resume)
    print('Training finished. Elapsed Time:', str(round((time.time()-start_time)/60,2)), 'min')
    
    
    #%% testing
    # # load model from best checkpoint if available otherwise last checkpoint is loaded automatically
    ckpt_path = trainer.checkpoint_callback.best_model_path # trainer.checkpoint_callback.last_model_path
    print('ckpt_path: ', ckpt_path)
    model = SegmentationModel.load_from_checkpoint(ckpt_path).to(device)
    model.eval()
    trainer.test(model=model, dataloaders=dataModule)
    wandb.finish()
    
    # # all class names and colors in order of index
    class_names = [cfg.data.classes[i].name for i in sorted(cfg.data.classes.keys(), key=int)]
    class_colors = [cfg.data.classes[i].color for i in sorted(cfg.data.classes.keys(), key=int)]
    
    
    #%% dataset-wise evaluation
    test_loader = dataModule.test_dataloader()
    metrics_dir = make_dir(cfg.log_path,'quantitative')
    
    # # # Classic Top-1
    metrics_dir_top_1 = make_dir(metrics_dir, 'top_1')
    metrics = evaluate_model(model, test_loader, cfg.model.num_classes, device, custom_avg=cfg.data.class_groups, save_dir=metrics_dir_top_1)
    visualize_scores_per_class(shrink_dict(metrics['semseg'], ['F1','IoU','Precision','Recall']), os.path.join(metrics_dir_top_1, 'semseg_scores.png'), class_names)
    visualize_save_confusions(metrics['conf'],metrics_dir_top_1,display_labels=class_names)
    print(f"Avg class-wise F1 scores - Top1Acc:  {np.array2string(metrics['semseg']['F1'], formatter={'float_kind': lambda x: f'{x:.3f}'})}")
    
    
    #%% sample-wise evaluation and qualitative visualization (pick-data)
    pick_loader = dataModule.pick_dataloader()
    qual_dir = make_dir(cfg.log_path,'qualitative')

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
