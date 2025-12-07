from torch.utils.data import DataLoader, WeightedRandomSampler
import pytorch_lightning as pl

from datasets.semseg_dataset import SegmentationDataset
from utils.utils import ensure_list_values

'''This file creates the SegmentationDataModule class (Lightning DataModule)

Handles:
- creating train/val/test/pick datasets
- applying different transforms per split
- constructing the respective DataLoaders
'''

class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, image_dirs, num_classes, transform_train, transform_val, transform_test, weightedSampling=False, value_mapping=None, ignore_index=-1, load_label=None, batch_size=16, num_workers=4):
        super().__init__()
        self.image_dirs = ensure_list_values(image_dirs)
        self.num_classes = num_classes
        self.transform_train = transform_train
        self.transform_val = transform_val
        self.transform_test = transform_test
        self.weightedSampling = weightedSampling
        self.value_mapping = value_mapping
        self.ignore_index = ignore_index
        self.load_label = load_label
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            # Initialize Datasets
            self.train_dataset = SegmentationDataset(self.image_dirs.train, self.num_classes, transform=self.transform_train, weightedSampling=self.weightedSampling, value_mapping=self.value_mapping, ignore_index=self.ignore_index, load_label=self.load_label)
            self.val_dataset = SegmentationDataset(self.image_dirs.val, self.num_classes, transform=self.transform_val, value_mapping=self.value_mapping, ignore_index=self.ignore_index, load_label=self.load_label)
            self.data_dims = self.train_dataset[0]['image'].shape
            # Initialize WeightedRandomSampler (only for train dataset)
            if self.weightedSampling:
                self.sampler = WeightedRandomSampler(self.train_dataset.sample_weights, num_samples=len(self.train_dataset), replacement=True)
        if stage == "test" or stage is None:
            # Initialize Datasets
            self.test_dataset = SegmentationDataset(self.image_dirs.test, self.num_classes, transform=self.transform_test, value_mapping=self.value_mapping, ignore_index=self.ignore_index, load_label=self.load_label)
            self.pick_dataset = SegmentationDataset(self.image_dirs.pick, self.num_classes, transform=self.transform_test, value_mapping=self.value_mapping, ignore_index=self.ignore_index, load_label=self.load_label)
            self.data_dims = self.test_dataset[0]['image'].shape

    # you could each add: multiprocessing_context='fork', persistent_workers=True, timeout=120
    def train_dataloader(self):
        if self.weightedSampling:
            return DataLoader(self.train_dataset, sampler=self.sampler, batch_size=self.batch_size, num_workers=self.num_workers) # shuffling automatically in sampler
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def pick_dataloader(self):
        return DataLoader(self.pick_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
