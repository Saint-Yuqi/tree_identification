import os
import rasterio
import numpy as np
import warnings

from torch.utils.data import Dataset
from omegaconf import ListConfig
from rasterio.errors import NotGeoreferencedWarning

from utils.utils import convert_mask_to_target, extract_dataset_name

warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)


class SegmentationDataset(Dataset):
    def __init__(self, image_dirs, num_classes, transform=None, geo=None, weightedSampling=None, value_mapping=None, ignore_index=-1, load_label=None):
        self.image_dirs = image_dirs if isinstance(image_dirs, ListConfig) or isinstance(image_dirs, list) else [image_dirs]
        self.num_classes = num_classes
        self.transform = transform
        self.geo = geo
        self.weightedSampling = weightedSampling
        self.value_mapping = value_mapping
        self.ignore_index = ignore_index
        self.load_label = load_label # can be 'multi-class', 'multi-label' or None
        
        # # Collect all image files and their corresponding mask files
        self.image_files = []
        self.mask_files = []
        for image_dir in self.image_dirs:
            images_subdir = os.path.join(image_dir, "images")
            for root, _, files in sorted(os.walk(images_subdir)):
                self.image_files.extend(sorted([os.path.join(root, f) for f in files if f.lower().endswith(('.tif', '.png', '.jpg'))]))
            masks_subdir = os.path.join(image_dir, "masks")
            if not os.path.isdir(masks_subdir):
                masks_subdir = os.path.join(image_dir, "labels")
            for root, _, files in sorted(os.walk(masks_subdir)):
                self.mask_files.extend(sorted([os.path.join(root, f) for f in files if f.lower().endswith(('.tif', '.png'))]))
        
        # # Ensure the number of images and masks are the same
        assert len(self.image_files) == len(self.mask_files), "Number of images and masks must be the same."
        
        # # Calculate class frequencies and sample weights
        if weightedSampling:
            self.class_presence = self.calculate_class_presence()
            self.sample_weights = self.calculate_sample_weights()
    
    def calculate_class_presence(self):
        # # Load mask and additional infos (e.g. canopy) based on file type 
        # # Set selected annotations (e.g. uncertain labels, or plants not visible on orthophotos but in the field (canopy=1)) to ignore_index
        class_presence = []
        for mask_path in self.mask_files:
            with rasterio.open(mask_path) as src:
                mask = src.read(1).astype(np.int16)
            
            # # Modify mask according to value_mapping
            dataset_name = extract_dataset_name(mask_path, self.value_mapping.keys())
            mapping_dict = self.value_mapping.get(dataset_name)
            if mapping_dict:
                for old_value, new_value in mapping_dict.items():
                    mask[mask == int(old_value)] = new_value
                
            # # calculate the presence of each class in every mask file and return it as list of dicts
            unique_classes = np.unique(mask)
            presence = {class_id: (1 if class_id in unique_classes else 0) for class_id in range(self.num_classes)}
            class_presence.append(presence)
    
        return class_presence
                
    def calculate_sample_weights(self):
        # # Calculate total occurrence of each class in the entire dataset
        total_class_presence = {key: 0 for key in range(self.num_classes)}
        for presence in self.class_presence:
            for key in total_class_presence:
                total_class_presence[key] += presence.get(key, 0)
                
        # # Fake a class presence in case of total absence to avoid divsion by 0
        total_class_presence = {key: (1 if value == 0 else value) for key, value in total_class_presence.items()}

        # # Calculate for each sample a weight, which is inversely proportional to presence frequency in the entire dataset, and return as a list
        sample_weights = []
        for presence in self.class_presence:
            sample_weight = sum(1.0 / total_class_presence[class_id] * presence[class_id] for class_id in range(self.num_classes))
            sample_weight = np.nan_to_num(sample_weight, nan=0.0)
            sample_weights.append(sample_weight)
        
        return sample_weights

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # # Load image (and meta infos)
        try:
            with rasterio.open(self.image_files[idx]) as src:
                image = src.read([1, 2, 3])
                image = np.stack(image, axis=-1)
                if self.geo:
                    meta = src.meta
                else:
                    meta = {}
        except rasterio.errors.RasterioIOError as e:
            print(f"Failed to read image at index {idx}: {self.image_files[idx]}")
            raise e  # R
        
        if image.dtype !='uint8':
            print(self.image_files[idx], image.dtype)

        # # Load mask
        try:
            with rasterio.open(self.mask_files[idx]) as src:
                mask = src.read(1).astype(np.int64)
        except rasterio.errors.RasterioIOError as e:
            print(f"Failed to read mask at index {idx}: {self.mask_files[idx]}")
            raise e  # R
            
        # # Modify mask according to value_mapping
        dataset_name = extract_dataset_name(self.mask_files[idx], self.value_mapping.keys())
        mapping_dict = self.value_mapping.get(dataset_name)
        if mapping_dict:
            for old_value, new_value in mapping_dict.items():
                mask[mask == int(old_value)] = new_value
        
        # # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        sample = {'image': image,
                  'mask': mask.long(),
                  'meta': meta,
                  'name': self.image_files[idx],
                  }
            
        if self.load_label:
            sample['label'] = convert_mask_to_target(mask, self.num_classes, self.ignore_index, self.load_label)
        
        return sample