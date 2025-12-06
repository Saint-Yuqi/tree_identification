import os
import json
import rasterio


"""
This file can be used to change the labels
input: 
- directory that contains labels (masks) in .png or .tif format
- mapping file (old to new label) (.json)

output: new labels for each image in input file (masks) in same format as inputs
 """

#specify input/output directory an mapping
input_dir = "/zfs/ai4good/datasets/tree/TreeAI/12_RGB_both/pick/labels_from_Bbox"
output_dir = "/zfs/ai4good/datasets/tree/TreeAI/12_RGB_both/pick/labels_Bbox_changed"
mapping_file = "label_mapping.json"
dataset_name= "dataset"

os.makedirs(output_dir, exist_ok=True)

# load mapping
with open(mapping_file, "r") as f:
    value_mapping = json.load(f)


#change the labels
def correct_mask(mask, mapping):
    mask_corrected = mask.copy()
    for old, new in mapping.items():
        mask_corrected[mask == int(old)] = int(new)
    return mask_corrected


#change labels for each image
for root, _, files in os.walk(input_dir):
    for file in files:
        if not file.lower().endswith((".png", ".tif")):
            print("file-ending..... strangeee")
            continue

        input_path = os.path.join(root, file)
        rel_path = os.path.relpath(input_path, input_dir)
        output_path = os.path.join(output_dir, rel_path)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with rasterio.open(input_path) as src:
            mask = src.read(1)
            meta = src.meta.copy()
            meta.pop("transform", None)
            meta.pop("crs", None)
            meta.pop("nodata", None)


        # determine which mapping to use
        mapping = value_mapping[dataset_name]

        mask_corrected = correct_mask(mask, mapping)

        #save new file
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(mask_corrected, 1)

print("Finished creating corrected labels.")