import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torchmetrics import Metric
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassJaccardIndex, MulticlassAUROC
from torch_prototypes.modules.prototypical_network import LearntPrototypes
from torch_prototypes.metrics.distortion import DistortionLoss
from losses_and_metrics.loss_functions import HierarchicalLoss, ProtoHierarchicalLoss
from losses_and_metrics.metrics import AverageCostMetric
from utils.scheduler_utils import get_scheduler


class SegmentationModel(pl.LightningModule):
    def __init__(self, model, encoder_name, img_size, num_classes, learning_rate, ignore_index=-1, optimizer='AdamW', lr_scheduler=None, loss='CE', weight=None, patch_2_img_size=False, d_matrix=None, d_matrix_eval=None, lossratio=3):
        super().__init__()
        self.save_hyperparameters()
        

        # # Model architecture
        if model=='Unet':
            self.model = smp.Unet(encoder_name=encoder_name, classes=num_classes)
        elif model=='Unet++':
            self.model = smp.UnetPlusPlus(encoder_name=encoder_name, classes=num_classes)
        elif model=='DeepLabV3+':
            self.model = smp.DeepLabV3Plus(encoder_name=encoder_name, classes=num_classes)
        else:
            raise ValueError('Invalid model specified.') 
        
        # # Loss function
        if weight:
            weight = torch.tensor(weight)
        if loss=='CE':
            self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        elif loss=='BCE':
            self.criterion = nn.BCEWithLogitsLoss(weight=weight, ignore_index=ignore_index)
        elif loss=='Focal':
            self.criterion = torch.hub.load('adeelh/pytorch-multi-class-focal-loss', model='FocalLoss', alpha=weight/sum(weight), gamma=2, reduction='mean', force_reload=False)
        elif loss =='Hierarchical':
            if d_matrix == None:
                raise ValueError("No D_matrix")
            # Register inside loss; it will be moved with the module automatically
            self.criterion = HierarchicalLoss(d_matrix, ignore_index=ignore_index)
        elif loss =='HierarchicalCE':
            if d_matrix == None:
                raise ValueError("No D_matrix")
            self.lossratio=lossratio
            self.criterion_ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
            self.criterion_hier= HierarchicalLoss(d_matrix, ignore_index=ignore_index)
            self.use_combined_loss = True    
        else:
            raise ValueError('Invalid loss function specified.')
        
        if d_matrix_eval == None:
                raise ValueError("No evaluation D_matrix")
        # # Metrics
        for split in ["train", "val", "test"]:
            for metric in ["acc", "iou", "f1", "auc","AHC"]:
                setattr(self, f"{split}_{metric}", {
                    "acc": MulticlassAccuracy(num_classes=num_classes, average='macro', ignore_index=ignore_index),
                    "iou": MulticlassJaccardIndex(num_classes=num_classes, average='macro', ignore_index=ignore_index),
                    "f1": MulticlassF1Score(num_classes=num_classes, average='macro', ignore_index=ignore_index),
                    "auc": MulticlassAUROC(num_classes=num_classes, average='macro', ignore_index=ignore_index, thresholds=5),
                    "AHC": AverageCostMetric(d_matrix_eval, ignore_index=ignore_index),
                }[metric])
        
    def set_patch_2_img_size(self, new_patch_2_img_size):
        self.hparams.patch_2_img_size = new_patch_2_img_size

    def configure_optimizers(self):
        if self.hparams.optimizer=='Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer=='AdamW':
            optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        else:
            raise ValueError('Invalid optimizer specified.')
        if self.hparams.lr_scheduler:
            scheduler = get_scheduler(optimizer, self.hparams.lr_scheduler)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}        
        return optimizer
    
    def load_pretrained_encoder_weights(self, weight_path):
        # # Load encoder weights (will only load matching keys)
        state_dict = torch.load(weight_path, map_location=self.device)
        missing, unexpected = self.model.encoder.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Missing keys when loading encoder: {missing}")
        if unexpected:
            print(f"Unexpected keys when loading encoder: {unexpected}")
    
    def _step(self, batch, stage):
        images = batch['image']
        masks = batch['mask']
        logits = self._forward(images) #torch.Tensor of shape [B,C,H,W] one logit per class

        if getattr(self, 'use_combined_loss', False):
            loss_ce = self.criterion_ce(logits, masks.long())
            loss_hier = self.criterion_hier(logits, masks.long())
            #alpha, beta = 0.3, 1  # weights to be tuned!!!!
            #loss = alpha * loss_ce + beta * loss_hier
            loss = loss_ce + self.lossratio * loss_hier
            self.log(f'{stage}_loss_ce', loss_ce, on_step=stage=='train', on_epoch=True, prog_bar=False, batch_size=images.shape[0])
            self.log(f'{stage}_loss_hier', loss_hier, on_step=stage=='train', on_epoch=True, prog_bar=False, batch_size=images.shape[0])
        else:
            loss = self.criterion(logits, masks.long())
       
        preds = torch.argmax(logits, dim=1) #torch.Tensor shape [B,H,W]: integer per pixel
        self.log(f'{stage}_loss', loss, on_step=stage=='train', on_epoch=True, prog_bar=True, batch_size=images.shape[0])
        getattr(self, f"{stage}_acc").update(preds.detach(), masks.detach())
        getattr(self, f"{stage}_iou").update(preds.detach(), masks.detach())
        getattr(self, f"{stage}_f1").update(preds.detach(), masks.detach())
        getattr(self, f"{stage}_auc").update(logits.detach(), masks.detach().long())
        getattr(self,f"{stage}_AHC").update(preds.detach(),masks.detach())

        return loss
    
    def _epoch_end(self, stage):
        self.log_dict({
            f"{stage}_acc": getattr(self, f"{stage}_acc").compute(),
            f"{stage}_iou": getattr(self, f"{stage}_iou").compute(),
            f"{stage}_f1": getattr(self, f"{stage}_f1").compute(),
            f"{stage}_auc": getattr(self, f"{stage}_auc").compute(),
            f"{stage}_AHC": getattr(self, f"{stage}_AHC").compute()
        }, prog_bar=True)
        getattr(self, f"{stage}_acc").reset()
        getattr(self, f"{stage}_iou").reset()
        getattr(self, f"{stage}_f1").reset()
        getattr(self, f"{stage}_auc").reset()
        getattr(self, f"{stage}_AHC").reset()
        
    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")
    
    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")
    
    def on_train_epoch_end(self):
        self._epoch_end("train")

    def on_validation_epoch_end(self):
        self._epoch_end("val")

    def on_test_epoch_end(self):
        self._epoch_end("test")    
    
    def _forward_with_patches(self, x):
        batch_size, channels, img_size, _ = x.shape
        patches, num_patches = patch_image_to_batch(x, self.hparams.img_size)
        output_patches = self.model(patches)
        outputs = merge_output_patches_to_image(output_patches, num_patches, self.hparams.img_size, batch_size, img_size)
        return outputs

    # # Internal (train/val/test) forward w/o final activation, because its inclduded in loss
    def _forward(self, x):
        if x.shape[-1] != self.hparams.img_size and self.hparams.patch_2_img_size:
            return self._forward_with_patches(x)
        return self.model(x)
    
    # # Inference forward with final activations included
    def forward(self, x):
        logits = self._forward(x)
        if self.hparams.loss in {'CE', 'Focal', 'Hierarchical', 'HierarchicalCE'}:
            return torch.softmax(logits, dim=1)
        elif self.hparams.loss=='BCE':
            return torch.sigmoid(logits)

     


class ProtoSegModel(pl.LightningModule):
    def __init__(self, model, encoder_name, img_size, num_classes, learning_rate, loss='CE', ignore_index=-1, weight=None, optimizer='AdamW', lr_scheduler=None, patch_2_img_size=False, dist='euclidean', prototype_init=None, d_matrix = None, d_matrix_eval=None, lambda_d = 0.1 ):
        super().__init__()
        self.save_hyperparameters()
        embedding_dim = 16
        decoder_channels = (256, 128, 64, 32, 16) #from the documentation...

        if model=='Unet':
            self.backbone = smp.Unet(encoder_name=encoder_name, classes=num_classes)
            self.backbone.segmentation_head = nn.Conv2d(in_channels=decoder_channels[-1] ,out_channels=embedding_dim, kernel_size = 1)
        else:
            raise ValueError('Model invalid')
       
        self.prototype_layer = LearntPrototypes(model=lambda x: x, n_prototypes=num_classes, embedding_dim=embedding_dim, dist=dist, prototypes=prototype_init)
        
        if d_matrix != None:
            self.disto_loss = DistortionLoss(D=d_matrix)
            self.lambda_d = lambda_d  # weight for distortion loss
        else:
            self.disto_loss = None
            self.lambda_d = 0.0

        self.ignore_index = ignore_index
        if weight:
            weight = torch.tensor(weight)
        
        if loss == 'CE':
            self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        
        elif loss == 'ProtoHierarchical':
            self.criterion_protohier = ProtoHierarchicalLoss(ignore_index = ignore_index)
        
        elif loss == 'ProtoHierarchicalCE':
            self.criterion_protohier = ProtoHierarchicalLoss(ignore_index=ignore_index)
            self.use_combined_loss = True 
        else:
            raise ValueError("Unsupported loss type")

        if d_matrix_eval == None:
                raise ValueError("No evaluation D_matrix")
                
        for split in ["train", "val", "test"]:
            for metric in ["acc", "iou", "f1", "auc", "AHC"]:
                setattr(self, f"{split}_{metric}", {
                    "acc": MulticlassAccuracy(num_classes=num_classes, average='macro', ignore_index=ignore_index),
                    "iou": MulticlassJaccardIndex(num_classes=num_classes, average='macro', ignore_index=ignore_index),
                    "f1": MulticlassF1Score(num_classes=num_classes, average='macro', ignore_index=ignore_index),
                    "auc": MulticlassAUROC(num_classes=num_classes, average='macro', ignore_index=ignore_index, thresholds=5),
                    "AHC": AverageCostMetric(d_matrix_eval,ignore_index=ignore_index),
                }[metric])

    def set_patch_2_img_size(self, new_patch_2_img_size):
        self.hparams.patch_2_img_size = new_patch_2_img_size

    def configure_optimizers(self):
        if self.hparams.optimizer=='Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer=='AdamW':
            optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        else:
            raise ValueError('Invalid optimizer specified.')
        if self.hparams.lr_scheduler:
            scheduler = get_scheduler(optimizer, self.hparams.lr_scheduler)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}        
        return optimizer


    def _step(self, batch, stage):
        images = batch['image']
        masks = batch['mask']
        neg_dists = self._forward(images) #neg. distances
        logits = torch.softmax(neg_dists, dim=1) #probabilities

        if getattr(self, 'use_combined_loss', False):
            loss_protohier = self.criterion_protohier(neg_dists, masks.long())
            self.disto_loss.D = self.disto_loss.D.to(self.prototype_layer.prototypes.device)
            loss_d = self.disto_loss(self.prototype_layer.prototypes)
            loss = loss_protohier + self.lambda_d * loss_d
            self.log(f'{stage}_loss_protohier', loss_protohier, on_step=stage=='train', on_epoch=True, prog_bar=False, batch_size=images.shape[0])
            self.log(f'{stage}_loss_disto', loss_d, on_step=stage=='train', on_epoch=True, prog_bar=False, batch_size=images.shape[0])
        
        elif self.hparams.loss =='CE':
            loss = self.criterion(neg_dists, masks.long())
            self.log(f'{stage}_loss_ce', loss, on_step=stage=='train', on_epoch=True, prog_bar=False, batch_size=images.shape[0])

        elif self.hparams.loss == 'ProtoHierarchical':
            loss = self.criterion(neg_dists, masks.long())
            self.log(f'{stage}_loss_protohier', loss, on_step=stage=='train', on_epoch=True, prog_bar=False, batch_size=images.shape[0])
        
        else:
            raise ValueError('Problems with loss in _step')

        preds = torch.argmax(logits, dim=1) #predictions
        self.log(f'{stage}_loss', loss, on_step=stage=='train', on_epoch=True, prog_bar=True, batch_size=images.shape[0])
        getattr(self, f"{stage}_acc").update(preds.detach(), masks.detach()) #update() is a function of the instance
        getattr(self, f"{stage}_iou").update(preds.detach(), masks.detach())
        getattr(self, f"{stage}_f1").update(preds.detach(), masks.detach())
        getattr(self, f"{stage}_auc").update(logits.detach(), masks.detach().long())
        getattr(self, f"{stage}_AHC").update(preds.detach(), masks.detach())  
        return loss


    def _epoch_end(self, stage):
        self.log_dict({
            f"{stage}_acc": getattr(self, f"{stage}_acc").compute(),
            f"{stage}_iou": getattr(self, f"{stage}_iou").compute(),
            f"{stage}_f1": getattr(self, f"{stage}_f1").compute(),
            f"{stage}_auc": getattr(self, f"{stage}_auc").compute(),
            f"{stage}_AHC": getattr(self, f"{stage}_AHC").compute(), 
        }, prog_bar=True)
        getattr(self, f"{stage}_acc").reset()
        getattr(self, f"{stage}_iou").reset()
        getattr(self, f"{stage}_f1").reset()
        getattr(self, f"{stage}_auc").reset()
        getattr(self,f"{stage}_AHC").reset() 
        
    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")
    
    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")
    
    def on_train_epoch_end(self):
        self._epoch_end("train")

    def on_validation_epoch_end(self):
        self._epoch_end("val")

    def on_test_epoch_end(self):
        self._epoch_end("test")    
    
    def _forward_with_patches(self, x):
        batch_size, channels, img_size, _ = x.shape
        patches, num_patches = patch_image_to_batch(x, self.hparams.img_size)
        embeddings = self.backbone(patches)    # (B, C_dec, H, W) at input res
        logits = self.prototype_layer(embeddings)  
        output_patches =  logits
        outputs = merge_output_patches_to_image(output_patches, num_patches, self.hparams.img_size, batch_size, img_size)
        return outputs

    # returns neg. distances between prototypes and points in embedding space
    def _forward(self, x):
        if x.shape[-1] != self.hparams.img_size and self.hparams.patch_2_img_size:
            return self._forward_with_patches(x)
        embeddings = self.backbone(x)   # (B, C_dec, H, W) at input res
        logits = self.prototype_layer(embeddings)  
        return logits # neg. distances betw. prototypes and points in embedding space

    
    # # Inference forward with final activations included
    def forward(self, x):
        logits = self._forward(x)
        if self.hparams.loss in {'CE', 'ProtoHierarchical', 'ProtoHierarchicalCE'}:
            return torch.softmax(logits, dim=1) #probabilities
        else:
            raise ValueError('Unsupportet Loss Type')
    


def patch_image_to_batch(image, patch_size):
    # # x shape: [batch_size, channels, height, width]
    _, channels, img_size, _ = image.shape

    # # Ensure the spatial dimensions are divisible by patch_size
    assert img_size % patch_size == 0, \
        "Img size must be divisible by patch_size."

    # # Patch the image using unfold
    patches = F.unfold(image, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))
    # # patches shape: [batch_size, channels * patch_size * patch_size, num_patches]

    # # Reshape to move patches to batch dimension
    num_patches = patches.shape[-1]
    patches = patches.permute(0, 2, 1) # [batch_size, num_patches, channels * patch_size * patch_size]
    patches = patches.reshape(-1, channels, patch_size, patch_size) # [batch_size * num_patches, channels, patch_size, patch_size]
    
    return patches, num_patches


def merge_output_patches_to_image(output_patches, num_patches, patch_size, batch_size, img_size):
    # # Reshape outputs back to [batch_size, channels * patch_size * patch_size, num_patches]
    output_patches = output_patches.reshape(batch_size, num_patches, -1)  # [batch_size, num_patches, channels * patch_size * patch_size]
    output_patches = output_patches.permute(0, 2, 1) # [batch_size, channels * patch_size * patch_size, num_patches]

    # # Fold the patches back to the original img size
    outputs = F.fold(output_patches, output_size=(img_size, img_size), kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))

    return outputs

