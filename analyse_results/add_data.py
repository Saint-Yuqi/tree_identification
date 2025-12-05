import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
- compare metrics of outcomes depending on how many new species got created.

TODO: depending on how many data are in there.
'''

#-------------------
#load data
#-------------------
BL_model = "/home/c/shursc/code/tree_identification/lightning_logs/20251119_093300.602560_Unet_resnet50_CE_320_62_wrs_True/quantitative/top_1/metrics_semseg.json"
model_add_data = "/home/c/shursc/code/tree_identification/lightning_logs/20251124_153621.661707_Unet_resnet50_CE_320_62_wrs_True/quantitative/top_1/metrics_semseg.json"

with open(BL_model, "r") as f:
    BL_model = json.load(f)

with open(model_add_data, "r") as f:
    model_add_data = json.load(f)

F1_BL = np.array(BL_model["IoU"], dtype=float)
F1_add_data = np.array(model_add_data["IoU"], dtype=float)

bbox_per_class = "/zfs/ai4good/datasets/tree/TreeAI/12_RGB_both/bbox_counts_train_folder.csv"
df_bbox_per_class = pd.read_csv(bbox_per_class)
bbox_map = dict(zip(df_bbox_per_class["class_id"], df_bbox_per_class["count"]))


tree_presence_original= "/home/c/shursc/code/tree_identification/analyse_data/tree_statistics_12.csv"
df_original_statistics = pd.read_csv(tree_presence_original)
idx_to_train_rel_pix = dict(zip(df_original_statistics["ID"], df_original_statistics["train_rel_pix_pres"]))
idx_to_train_rel_im = dict(zip(df_original_statistics["ID"], df_original_statistics["train_rel_im_pres"]))




#create table

tabular_F1 = []
for cls_id in range(len(F1_BL)):
    if cls_id not in bbox_map:
        continue

    F1_base = F1_BL[cls_id]
    F1_new = F1_add_data[cls_id]

    if np.isnan(F1_base) or np.isnan(F1_new):
        print(f"{cls_id} has NaN value.")
        continue
    
    abs_change = F1_new - F1_base

    rel_change = abs_change / F1_base if F1_base !=0 else np.nan

    tabular_F1.append({
        "class_id": cls_id,
        "bbox_count": bbox_map[cls_id],
        "original_presence": idx_to_train_rel_pix[cls_id],
        "original_im_pres": idx_to_train_rel_im[cls_id],
        "ratio": bbox_map[cls_id]/idx_to_train_rel_pix[cls_id],
        "F1_base": F1_base,
        "F1_new": F1_new,
        "F1_abs_change": abs_change,
        "F1_rel_change": rel_change,
        
    })

df =pd.DataFrame(tabular_F1)

df = df.sort_values(by="bbox_count",ascending=False)

print(df)


x_labels = df["class_id"].tolist()                  
x_positions = range(len(df))                          
y = df["F1_abs_change"]

plt.figure(figsize=(12, 5))
plt.bar(x_positions, y)
plt.xlabel("class_id")
plt.ylabel("F1_abs_change")
plt.title("F1 absolute change per class (sorted by additional trees)")
plt.xticks(x_positions, x_labels, rotation=90)
plt.tight_layout()
plt.savefig("F1_abs_change_vs_class_id_sorted_bboxcount.png", dpi=300)
plt.close()



x = df["ratio"]
y = df["F1_abs_change"]

plt.figure(figsize=(8, 6))
plt.scatter(x, y)

plt.xlabel("New Bbox / original # pixels")
plt.ylabel("F1_abs_change")
plt.title("F1 absolute change vs. new Bboxes/original presence")

plt.tight_layout()
plt.savefig("F1_abs_change_vs_ratio.png", dpi=300)
plt.close()
