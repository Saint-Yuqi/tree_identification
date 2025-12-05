import numpy as np
import rasterio
import yaml
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import time
import pandas as pd


#specify which data you want to consider: SemSem (12 or 34)
number= 34


with open("../configs/data/treeAI_classes.yaml") as f:
    class_cfg = yaml.safe_load(f)

id_to_name = {int(k): v["name"] for k, v in class_cfg["classes"].items()}



def analyze_folder(folder_path):
    label_folder = Path(folder_path)
    label_files = list(label_folder.glob("*.png"))

    total_images = len(label_files)

    image_presence = Counter()
    pixel_presence = Counter()

    
    for label_file in label_files:
        with rasterio.open(label_file) as src:
            labels = src.read(1).astype(np.int64)

        for cls_id in np.unique(labels):
            image_presence[int(cls_id)] += 1
        
        unique, counts = np.unique(labels, return_counts=True)
        pixel_presence.update({int(k): int(v) for k, v in dict(zip(unique, counts)).items()})

    total_pixels = sum(pixel_presence.values())

    return total_images, image_presence, pixel_presence, total_pixels


#############################################################################################

if number == 12:
    labeled = "fL"
elif number == 34:
    labeled = "pL"
else:
    print("check your data")

folders = {
    "train": f"/zfs/ai4good/datasets/tree/TreeAI/{number}_RGB_SemSegm_640_{labeled}/train/labels",
    "val":   f"/zfs/ai4good/datasets/tree/TreeAI/{number}_RGB_SemSegm_640_{labeled}/val/labels",
    "test":  f"/zfs/ai4good/datasets/tree/TreeAI/{number}_RGB_SemSegm_640_{labeled}/test/labels",
    "pick":  f"/zfs/ai4good/datasets/tree/TreeAI/{number}_RGB_SemSegm_640_{labeled}/pick/labels"
}
print(f"Data: {number}_RGB_SenSegm_640_{labeled}")



data_dict = {"Tree Name": [], "ID": [], "train_abs_im_pres": [], "train_rel_im_pres":[], "test_abs_im_pres":[], "test_rel_im_pres":[],"val_abs_im_pres": [], "val_rel_im_pres":[], "pick_abs_im_pres":[], "pick_rel_im_pres":[] , "train_abs_pix_pres": [], "train_rel_pix_pres":[], "test_abs_pix_pres":[], "test_rel_pix_pres":[],"val_abs_pix_pres": [], "val_rel_pix_pres":[], "pick_abs_pix_pres":[], "pick_rel_pix_pres":[]}

all_ids = sorted(id_to_name.keys())

for cls_id in all_ids:
    data_dict["Tree Name"].append(id_to_name[cls_id])
    data_dict["ID"].append(cls_id)

for key in data_dict.keys():
    if key not in ["Tree Name", "ID"]:
        data_dict[key] = [0]*len(all_ids)


for name, folder in folders.items():
    total_images, image_presence, pixel_presence, total_pixels = analyze_folder(folder)
    if total_images > 0 and total_pixels > 0:
        rel_image_presence = {key: val/total_images for key, val in image_presence.items()}
        rel_pixel_presence = {key: val/total_pixels for key, val in pixel_presence.items()}
    else:
        print("non-positive pixel or images.")

    for idx, cls_id in enumerate(all_ids):
        abs_im = image_presence.get(cls_id, 0)
        rel_im = rel_image_presence.get(cls_id, 0)
        abs_pix = pixel_presence.get(cls_id,0)
        rel_pix = rel_pixel_presence.get(cls_id,0)

        data_dict[f"{name}_abs_im_pres"][idx] = abs_im
        data_dict[f"{name}_rel_im_pres"][idx] = rel_im
        data_dict[f"{name}_abs_pix_pres"][idx] = abs_pix
        data_dict[f"{name}_rel_pix_pres"][idx] = rel_pix

        

df = pd.DataFrame(data_dict)
df.to_csv(f"tree_statistics_{number}.csv", index=False)
df.to_excel(f"tree_statistics_{number}.xlsx", index=False)

print(f"Saved tree_statistics_{number}.xlsx")








    
'''   

start = time.time()
total_images, image_presence, pixel_presence, total_pixels = analyze_folder(mask_folder_train)
end = time.time()

print(f"Time taken: {end - start} seconds")

print("Total images: ", total_images)
print("Image presence: ", image_presence)
print("Pixel presence: ", pixel_presence)
print("Total pixels: ", total_pixels)
'''
