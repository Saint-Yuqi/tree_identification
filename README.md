
# 🌳 TreeAI Segmentation
A deep learning pipeline for **pixel-wise tree segmentation** from aerial imagery enabling both species classification and crown delineation.
This repository is intended as a **starting point for further developments in tree segmentation** as part of the **UZH Master's course AI4Good**.

## ✨ Motivation
Tree species segmentation plays a crucial role in **biodiversity monitoring**, **forest management**, and **climate change research**.  
Accurate mapping of tree species enables better decision-making for conservation efforts and sustainable forestry.

## 📊 Dataset
We use the **TreeAI dataset**, a **global collection of tree species annotations and high-resolution imagery**.  

- Contains **large-scale samples across diverse ecosystems**.  
- Published on **Zenodo**: [TreeAI dataset](https://zenodo.org/records/15351054)
- Further details:  
  - [EGU 2025 abstract](https://meetingorganizer.copernicus.org/EGU25/EGU25-18117.html)  
  - [SmartForest 2025 abstract](https://smartforest.ai/wp-content/uploads/2025/02/SmartForest-2025-Abstract-TreeAI-a-global-database-for-tree-species-annotations-and-high-resolution-imagery.pdf)

The default storage location originating from this repo is `../data/TreeAI/`. See under Project Structure for more information. A test set of the data for the AI4Good course is located on the iMath server.

## ⚙️ Environment Setup
Clone this repo. Install [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-2). Create a new conda environment **treeAI** and install the following dependencies. **Weights & Biases** (account needed) is the default for logging, but other tools work similarly.

```bash
conda create -n treeAI python=3.12
conda activate treeAI

# Base packages: numpy, pytorch, lightning
conda install conda-forge::numpy
pip3 install torch torchvision
python -m pip install lightning

# Visualization
conda install conda-forge::matplotlib

# Logging
conda install conda-forge::wandb

# Configs
conda install conda-forge::hydra-core

# Data
conda install conda-forge::rasterio
conda install conda-forge::albumentations
conda install conda-forge::scikit-learn

# Segmentation models
pip install segmentation-models-pytorch
```

## 🚀 Training & Testing
This repository uses Hydra for flexible experiment configuration. So model training can be just started with `train.py`. 

Corresponding configuration files are in `configs/train.yaml` with second level configs listed in the header of this file. Those second-level configs are grouped:

- **data**: `configs/data/treeAI.yaml`
- **model**: `configs/data/model_treeAI.yaml`

These files serve as the default configurations. The same structure applies for `test.py`, where at least the experiment name on which the evaluation is to take place must be specified in `test.yaml` or the command line.

All parameters can be overwritten from the command line. If multiple data or model parameters need to be changed simultaneously, it is often convenient to create a custom YAML file following the same structure as the default files. They can then be overwritten again in the command line.
```
python train.py data=data_treeAI_vernon model=model_UNet_resnet18
```

## 📂 Project Structure
```
data/
│── TreeAI
│ ├── 12_RGB_SemSegm_640_fl
│ │ ├── train                 # training data 
│ │ │ ├── images
│ │ │ ├── labels
│ │ ├── val                   # validation data 
│ │ │ ├── images
│ │ │ ├── labels
│ │ ├── test                  # test data 
│ │ │ ├── images
│ │ │ ├── labels
│ │ ├── pick                  # selected data for visualization
│ │ │ ├── images
│ │ │ ├── labels
│ ├── 34_RGB_SemSegm_640_pL
│ │ ├── ...
treeAI-segmentation/
│── train.py                  # training script
│── test.py                   # testing / evaluation
│── configs/                  # hydra configuration files
│── models/                   # model definitions
│── data/                     # dataset loading & preprocessing
│── utils/                    # helper functions
│── README.md                 # this file
```

