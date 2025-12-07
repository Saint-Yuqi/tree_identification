import numpy as np
import torch
import torch.nn.functional as F
import os
import re
from typing import Literal
from omegaconf import ListConfig, DictConfig

"""
Utility functions for data handling, normalisation, file traversal, target
conversion, and general helpers used across the segmentation pipeline.

Includes:
    - image denormalisation
    - recursive file discovery
    - folder creation
    - ndarray <-> list conversion
    - mask/target conversion (multilabel + multiclass)
    - dataset name extraction
    - list-wrapping of config directory paths
"""

# =============================================================================
# denormalize numpy image
def denormalize(image, mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225])):
    image = image.transpose((1, 2, 0))
    image = (image * std + mean)
    return np.clip(image, 0, 1)

# =============================================================================
# Denormalization class for tensors
class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        deNorm_tensor = tensor.clone()
        for t, m, s in zip(deNorm_tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return deNorm_tensor
    
# =============================================================================
def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

# =============================================================================
def getListOfImgFiles(dirName):
    # run getListOfFiles
    allFiles = getListOfFiles(dirName)
    
    # filter img data
    for fichier in allFiles[:]: # im_names[:] makes a copy of im_names.
        if not(fichier.endswith(".png") or fichier.endswith(".jpg") or fichier.endswith(".tif")):
            allFiles.remove(fichier)
    
    return allFiles

# ============================================================================= 
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

# ============================================================================= 
def make_dir(directory,name):
    """ generates a folder if not exists """
    folder = os.path.join(directory,name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

def make_folder(directory,name):
    """ Alias for make_dir """
    return make_dir(directory, name)

# =============================================================================
def convert_ndarray_to_list(data):
    if isinstance(data, dict):
        return {key: convert_ndarray_to_list(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_ndarray_to_list(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data

# =============================================================================
def shrink_dict(original_dict, keep_keys):
    new_dict = {key: original_dict[key] for key in keep_keys if key in original_dict}
    return new_dict

# =============================================================================    
def one_hot_encode(image: torch.Tensor, num_classes: int):
    '''
    Convert an image of shape [B, W, H] to one-hot encoding of shape [B, C, W, H].

    Args:
        image (torch.Tensor): Input tensor of shape [B, W, H] with integer values.
        num_classes (int): Number of classes (C) for one-hot encoding.

    Returns:
        torch.Tensor: One-hot encoded tensor of shape [B, C, W, H].
    '''
    # Ensure the image tensor is of integer type
    image = image.long()
    
    # Apply one-hot encoding and reshape
    one_hot = F.one_hot(image, num_classes=num_classes)  # Shape: [B, W, H, C]
    one_hot = one_hot.permute(0, 3, 1, 2)  # Rearrange to [B, C, W, H]
    
    return one_hot.float()

# =============================================================================    
def convert_mask_to_target(
    mask: torch.Tensor,             # shape: (H, W), dtype: torch.long
    num_classes: int,               # total number of classes including background (0)
    ignore_index: int = -1,              # ignore_index (not needed here -> set to background)
    mode: Literal["multilabel", "multiclass"] = "multilabel"
    ) -> torch.Tensor:
    """
    Convert a single mask to a binary class vector of shape (num_classes,).

    In both modes:
      - background (class 0) is included only if no other class is present
      - returned tensor has 1s for present class(es), 0s elsewhere

    Args:
        mask: Long tensor of shape (H, W), values in [0, num_classes - 1]
        num_classes: Total number of classes (incl. background = 0)
        mode: "multi-label" or "multi-class"

    Returns:
        Tensor of shape (num_classes,) with 0s and 1s
    """
    flat_mask = mask.view(-1)
    flat_mask[flat_mask == ignore_index] = 0 
    hist = torch.bincount(flat_mask, minlength=num_classes)
    target = torch.zeros(num_classes, dtype=torch.float32, device=mask.device)

    if mode == "multilabel":
        target[hist > 0] = 1
        if target[1:].sum() > 0:
            target[0] = 0  # suppress background if any other class exists

    elif mode == "multiclass":
        hist[0] = 0  # ignore background for majority
        if hist.sum() == 0:
            target[0] = 1  # only background
        else:
            majority_class = torch.argmax(hist)
            target[majority_class] = 1

    return target

# =============================================================================    
def extract_dataset_name(file_path, known_dataset_names):
    path_parts = file_path.split(os.sep)
    for part in reversed(path_parts):  # reverse to match from inner to outer
        if part in known_dataset_names:
            return part
    raise ValueError(f"No known dataset name found in path: {file_path}")

# =============================================================================    
def ensure_list_values(dirs_dict):
    # Iterate over all keys in the dictionary
    for key, value in dirs_dict.items():
        # Check if the value is a dictionary, and recursively process it
        if isinstance(value, dict) or isinstance(value, DictConfig):
            ensure_list_values(value)
        else:
            # Apply the transformation: ensure the value is a list
            dirs_dict[key] = value if isinstance(value, ListConfig) or isinstance(value, list) else [value]
    return dirs_dict        
