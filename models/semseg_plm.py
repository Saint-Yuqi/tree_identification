import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassJaccardIndex, MulticlassAUROC

from utils.scheduler_utils import get_scheduler


class HierarchicalLoss(nn.Module):
    def __init__(self, distance_matrix: torch.Tensor, ignore_index: int = -100, reduction: str = 'mean'):
        """
        distance_matrix: Tensor of shape (num_classes, num_classes) with D[i, j] = distance(i, j)
        ignore_index: label to ignore in the loss (e.g., for unlabeled pixels)
        reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.register_buffer("D", distance_matrix.float())
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C, H, W) class logits per pixel (B: pct. per batch, C: 62 (nr. classes), H=W=32
        targets: (B, H, W) integer class labels: stores truth class

        """
         
        # Flatten spatial dimensions to compute per-pixel loss
        B, C, H, W = logits.shape
        probs = F.softmax(logits, dim=1)  # change model outputs to probabilities mit e^.. 
        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, C)  # (B, H, W, C) -> (N, C) N =B*H*C
        targets_flat = targets.reshape(-1)  # 1D vector (N,)
        
        #get rid of invalid indices
        valid_mask = targets_flat != self.ignore_index #should be considered in the loss (T vs. F) #for each pixel individually
        if not torch.any(valid_mask):
            return probs_flat.new_tensor(0.0, requires_grad=True)
        targets_valid = targets_flat[valid_mask] #remove pixels which are not used
        probs_valid = probs_flat[valid_mask]  # (N_valid, C)
        
        # calculate the loss:
        D_y = self.D[targets_valid]  # (N_valid, C) for each pixel individual the row in D
        losses = torch.sum(probs_valid * D_y, dim=1)  # (N_valid,), elementwise multiplication (dim 1)

        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses



class SegmentationModel(pl.LightningModule):
    def __init__(self, model, encoder_name, img_size, num_classes, learning_rate, ignore_index=-1, optimizer='AdamW', lr_scheduler=None, loss='CE', weight=None, patch_2_img_size=False, d_matrix=None):
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
            raise ValueError('Invalid model specified.') # ...but much more models are available at smp!
        
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
            self.criterion_ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
            self.criterion_hier= HierarchicalLoss(d_matrix, ignore_index=ignore_index)
            self.use_combined_loss = True    
        else:
            raise ValueError('Invalid loss function specified.')
        
        # # Metrics
        for split in ["train", "val", "test"]:
            for metric in ["acc", "iou", "f1", "auc"]:
                setattr(self, f"{split}_{metric}", {
                    "acc": MulticlassAccuracy(num_classes=num_classes, average='macro', ignore_index=ignore_index),
                    "iou": MulticlassJaccardIndex(num_classes=num_classes, average='macro', ignore_index=ignore_index),
                    "f1": MulticlassF1Score(num_classes=num_classes, average='macro', ignore_index=ignore_index),
                    "auc": MulticlassAUROC(num_classes=num_classes, average='macro', ignore_index=ignore_index, thresholds=5),
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
        logits = self._forward(images)

        if getattr(self, 'use_combined_loss', False):
            loss_ce = self.criterion_ce(logits, masks.long())
            loss_hier = self.criterion_hier(logits, masks.long())
            alpha, beta = 0.6, 0.4  # weights to be tuned!!!!
            loss = alpha * loss_ce + beta * loss_hier
            self.log(f'{stage}_loss_ce', loss_ce, on_step=stage=='train', on_epoch=True, prog_bar=False, batch_size=images.shape[0])
            self.log(f'{stage}_loss_hier', loss_hier, on_step=stage=='train', on_epoch=True, prog_bar=False, batch_size=images.shape[0])
        else:
            loss = self.criterion(logits, masks.long())

        #loss = self.criterion(logits, masks.long())
       
        preds = torch.argmax(logits, dim=1)
        self.log(f'{stage}_loss', loss, on_step=stage=='train', on_epoch=True, prog_bar=True, batch_size=images.shape[0])
        getattr(self, f"{stage}_acc").update(preds.detach(), masks.detach())
        getattr(self, f"{stage}_iou").update(preds.detach(), masks.detach())
        getattr(self, f"{stage}_f1").update(preds.detach(), masks.detach())
        getattr(self, f"{stage}_auc").update(logits.detach(), masks.detach().long())
        return loss
    
    def _epoch_end(self, stage):
        self.log_dict({
            f"{stage}_acc": getattr(self, f"{stage}_acc").compute(),
            f"{stage}_iou": getattr(self, f"{stage}_iou").compute(),
            f"{stage}_f1": getattr(self, f"{stage}_f1").compute(),
            f"{stage}_auc": getattr(self, f"{stage}_auc").compute(),
        }, prog_bar=True)
        getattr(self, f"{stage}_acc").reset()
        getattr(self, f"{stage}_iou").reset()
        getattr(self, f"{stage}_f1").reset()
        getattr(self, f"{stage}_auc").reset()
        
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
        if self.hparams.loss in {'CE', 'Focal', 'Hierarchical'}:
            return torch.softmax(logits, dim=1)
        elif self.hparams.loss=='BCE':
            return torch.sigmoid(logits)

#from chat:
'''
class SegmentationModelDistortion(SegmentationModel):
    def __init__(self, model, encoder_name, img_size, num_classes, learning_rate, ignore_index=-1, optimizer='AdamW', lr_scheduler=None, loss='CE', weight=None, patch_2_img_size=False, embedding_dim=16, num_prototypes_per_class=2, lambda_disto=1.0, D_tensor=None, use_proto_logits=False):
        super().__init__(model, encoder_name, img_size, num_classes, learning_rate, ignore_index=ignore_index, optimizer=optimizer, lr_scheduler=lr_scheduler, loss=loss, weight=weight, patch_2_img_size=patch_2_img_size)
        # Per-pixel embedding head on top of logits (1x1 conv maps class-logits to embedding space)
        self.embedding_dim = embedding_dim
        self.use_proto_logits = use_proto_logits
        self.emb_head = nn.Conv2d(num_classes, embedding_dim, kernel_size=1, bias=False)
        # Prototypes: either internal tensor [C, K, E] or external LearntPrototypes (one per class)
        self.num_prototypes_per_class = num_prototypes_per_class
        self.prototypes = nn.Parameter(torch.randn(num_classes, num_prototypes_per_class, embedding_dim) * 0.01)
        self.lp = None
        if self.use_proto_logits and TP_LearntPrototypes is not None and num_prototypes_per_class == 1:
            # Use torch_prototypes LearntPrototypes with one prototype per class
            # We pass embeddings directly via an Identity backbone
            self.lp = TP_LearntPrototypes(model=nn.Identity(), n_prototypes=num_classes, embedding_dim=embedding_dim, prototypes=None, device='cpu')
        # Distortion guidance weight
        self.lambda_disto = float(lambda_disto)
        if D_tensor is not None:
            self.register_buffer('D_metric', D_tensor)
        else:
            self.register_buffer('D_metric', None)

    def _compute_embeddings(self, x):
        # Get model logits then map to embedding space
        logits = super()._forward(x)
        embeddings = self.emb_head(logits)
        return logits, embeddings

    @staticmethod
    def _pairwise_class_distances(centroids):
        # centroids: [C, E] -> pairwise squared Euclidean distances [C, C]
        c2 = (centroids ** 2).sum(dim=1, keepdim=True)  # [C,1]
        dist2 = c2 + c2.t() - 2.0 * centroids @ centroids.t()
        dist2 = torch.clamp(dist2, min=0.0)
        return torch.sqrt(dist2 + 1e-12)

    def _distortion_loss(self):
        if self.D_metric is None:
            return None
        # Use class centroids (mean over prototypes)
        centroids = self.prototypes.mean(dim=1)  # [C, E]
        D_pred = self._pairwise_class_distances(centroids)  # [C, C]
        # Normalize both to comparable scales
        D_target = self.D_metric.to(D_pred.dtype)
        # Ensure shapes match number of classes
        C = centroids.shape[0]
        if D_target.shape[0] != C or D_target.shape[1] != C:
            # If mismatch, try to slice or return None
            if D_target.shape[0] >= C and D_target.shape[1] >= C:
                D_target = D_target[:C, :C]
            else:
                return None
        # Scale-invariant MSE by dividing each matrix by its mean (avoid degenerate zeros)
        mean_pred = torch.clamp(D_pred.mean(), min=1e-6)
        mean_tgt = torch.clamp(D_target.mean(), min=1e-6)
        D_pred_n = D_pred / mean_pred
        D_tgt_n = D_target / mean_tgt
        return F.mse_loss(D_pred_n, D_tgt_n)

    def _proto_class_logits(self, embeddings):
        # embeddings: [B, E, H, W]
        if self.lp is not None:
            # Use torch_prototypes: returns -dists of shape [B, C, H, W] when n_prototypes==num_classes
            return self.lp(embeddings)
        B, E, H, W = embeddings.shape
        C = self.prototypes.shape[0]
        K = self.prototypes.shape[1]
        # Flatten spatial
        emb_flat = embeddings.permute(0, 2, 3, 1).reshape(-1, E)  # [B*H*W, E]
        protos = self.prototypes.reshape(C * K, E)  # [C*K, E]
        # Compute squared distances: ||e - p||^2 = e^2 + p^2 - 2 e.p
        e2 = (emb_flat ** 2).sum(dim=1, keepdim=True)  # [N,1]
        p2 = (protos ** 2).sum(dim=1).unsqueeze(0)     # [1, C*K]
        dist2 = e2 + p2 - 2.0 * emb_flat @ protos.t()  # [N, C*K]
        dist2 = torch.clamp(dist2, min=0.0)
        # Reshape and aggregate per class
        dist2 = dist2.view(B * H * W, C, K)            # [N, C, K]
        # Use min distance to class prototypes; logits as negative distance
        min_dist2, _ = dist2.min(dim=2)                # [N, C]
        logits = (-min_dist2).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        return logits

    def _step(self, batch, stage):
        images = batch['image']
        masks = batch['mask']
        base_logits, embeddings = self._compute_embeddings(images)
        if self.use_proto_logits:
            logits = self._proto_class_logits(embeddings)
        else:
            logits = base_logits
        loss_ce = self.criterion(logits, masks.long())
        loss = loss_ce
        loss_disto = self._distortion_loss()
        if loss_disto is not None and self.lambda_disto > 0:
            loss = loss + self.lambda_disto * loss_disto
            self.log(f'{stage}_loss_disto', loss_disto, on_step=stage=='train', on_epoch=True, prog_bar=False, batch_size=images.shape[0])
        preds = torch.argmax(logits, dim=1)
        self.log(f'{stage}_loss', loss, on_step=stage=='train', on_epoch=True, prog_bar=True, batch_size=images.shape[0])
        getattr(self, f"{stage}_acc").update(preds.detach(), masks.detach())
        getattr(self, f"{stage}_iou").update(preds.detach(), masks.detach())
        getattr(self, f"{stage}_f1").update(preds.detach(), masks.detach())
        getattr(self, f"{stage}_auc").update(logits.detach(), masks.detach().long())
        return loss
'''

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
