from PIL import Image
import numpy as np

# Load image
image_link = "/home/c/shursc/data/TreeAI/12_RGB_SemSegm_640_fL/train/labels/000000002490.png"
img = Image.open(image_link)

# Convert to numpy array (keeps integer values)
mask = np.array(img)

print(mask.dtype)      # usually uint8 for labels 0–255
print(mask.shape)      # height × width
print(np.unique(mask)) # 

