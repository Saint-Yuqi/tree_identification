import os
import hashlib
from PIL import Image
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import shutil
import json
''' This file helps to compare Bbox and Mask labels between ObjDet and SemSegm Data
- which images are in both datasets?
- how do indices relate?

+ commmented out:
- creates new directories for shared and non-shared images
'''


def file_hash(path):
    """Return SHA256 hash of the file's binary content."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        # read in chunks to handle large images
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()



def get_hash_map(folders):
    """
    Returns:
        hash_map: dict {hash: [(foldername, filename, fullpath), ...]}
        count: total number of .png files
    """
    hash_map = {}
    count = 0
    for folder in folders:
        for fname in os.listdir(folder):
            if fname.lower().endswith(".png"):
                fullpath = os.path.join(folder, fname)
                try:
                    h = file_hash(fullpath)
                    hash_map.setdefault(h, []).append((folder, fname, fullpath))
                    count += 1
                except Exception as e:
                    print("Error reading", fullpath, e)
    return hash_map, count



# Read SemSegm mask (PNG)
def read_semseg_mask(image_path):
    label_path = image_path.replace("/images/", "/labels/")
    if not os.path.exists(label_path):
        return None
    mask = np.array(Image.open(label_path))
    return sorted(np.unique(mask).tolist())


# Read ObjDet label
def read_objdet_classes(image_path):
    """
    Reads ObjDet labels which may be YOLO .txt or XML (Pascal VOC/ArcGIS).
    Returns sorted unique class IDs.
    """
    # Candidate label paths
    txt_path = image_path.replace("/images/", "/labels/").replace(".png", ".txt")
    xml_path = image_path.replace("/images/", "/labels/").replace(".png", ".xml")

    # Case 1: TXT
    if os.path.exists(txt_path):
        classes = []
        with open(txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                cls_id = int(float(line.split()[0]))  # first value
                classes.append(cls_id)
        return sorted(list(set(classes)))

    # Case 2: XML label (test)
    if os.path.exists(xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        classes = []
        for obj in root.findall("object"):
            name_tag = obj.find("name")
            if name_tag is not None:
                # Class ID inside <name> tag (string → int)
                classes.append(int(name_tag.text))
        return sorted(list(set(classes)))
    # No label file found
    return None



# LOOKUP: HASH → folder (train/test/val)
def copy_non_overlapping_images(source_folder, dest_folder):
    """
    Copies all images from source_folder to dest_folder
    *except* those whose hash is in common_hashes.
    """
    for fname in os.listdir(source_folder):
        if not fname.lower().endswith(".png"):
            continue

        src_path = os.path.join(source_folder, fname)
        h = file_hash(src_path)

        # Skip images present in both datasets
        if h in common_hashes:
            continue

        # Copy the image
        shutil.copy2(src_path, os.path.join(dest_folder, fname))

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


A_folders = ["/zfs/ai4good/datasets/tree/TreeAI/12_RGB_SemSegm_640_fL/train/images", 
    "/zfs/ai4good/datasets/tree/TreeAI/12_RGB_SemSegm_640_fL/test/images", 
    "/zfs/ai4good/datasets/tree/TreeAI/12_RGB_SemSegm_640_fL/val/images", 
    "/zfs/ai4good/datasets/tree/TreeAI/12_RGB_SemSegm_640_fL/pick/images"]

B_train = "/zfs/ai4good/datasets/tree/TreeAI/12_RGB_ObjDet_640_fL/train/images"
B_test = "/zfs/ai4good/datasets/tree/TreeAI/12_RGB_ObjDet_640_fL/test/images"
B_val = "/zfs/ai4good/datasets/tree/TreeAI/12_RGB_ObjDet_640_fL/val/images"
B_folders = [B_train, B_test, B_val]

#build hash maps for identical images
map_A, count_A = get_hash_map(A_folders)
map_B, count_B = get_hash_map(B_folders)

common_hashes = set(map_A.keys()).intersection(map_B.keys())


print("Number of images in A:", count_A)
print("Number of images in B:", count_B)
print("Number of identical images:", len(common_hashes))



#print first identical pairs:
print("First 10 identical image pairs:")
printed = 0

for h in common_hashes:
    for folderA, fnameA, _ in map_A[h]:
        for folderB, fnameB, _ in map_B[h]:
            print(f"{folderA}: {fnameA}   <-->   {folderB}: {fnameB}")
            printed += 1
            if printed == 10:
                break
        if printed == 10:
            break
    if printed == 10:
        break


# Build table to compare indices
records = []

for h in common_hashes:
    for folderA, fnameA, fullA in map_A[h]:
        for folderB, fnameB, fullB in map_B[h]:

            # Label indices for SemSegm
            unique_A = read_semseg_mask(fullA)

            # Label indices for ObjDet (from txt)
            unique_B = read_objdet_classes(fullB)

            records.append({
                "SemSegm": fnameA,
                "ObjDet": fnameB,
                "IndexSemSegm": unique_A,
                "IndexObjDet": unique_B
            })

df = pd.DataFrame(records)
print(df.head(10))



# Filter for images for certain classes (ObjDet Index)

classnumber= 40
df_labelx = df[df["IndexObjDet"].apply(lambda lst: lst is not None and classnumber in lst)]

print(f"Number of identical images where ObjDet contains label {classnumber}:",
      len(df_labelx))

print(df_labelx.head(10))





######################################################################################################
# code from here just creates new folders and copies the corresponding labels to the correct places.
######################################################################################################

#create new folder with images, that are not already in SemSegm folder.
'''
new_root = "/zfs/ai4good/datasets/tree/TreeAI/12_RGB_ObjDet_new_masks"

new_train = os.path.join(new_root, "train/images")
new_test  = os.path.join(new_root, "test/images")
new_val   = os.path.join(new_root, "val/images")

for f in [new_train, new_test, new_val]:
    os.makedirs(f, exist_ok=True)

copy_non_overlapping_images(B_train, new_train)
copy_non_overlapping_images(B_test,  new_test)
copy_non_overlapping_images(B_val,   new_val)

print("Finished creating filtered ObjDet dataset.")'''


#copy labels to new folder:
'''
new_root = "/zfs/ai4good/datasets/tree/TreeAI/12_RGB_ObjDet_new_masks"
B_root = "/zfs/ai4good/student/shursc/treeAI/12_RGB_ObjDet_640_fL"

splits = ["train", "test", "val"]


for split in splits:
    # Paths
    new_images_dir = os.path.join(new_root, split, "images")
    new_labels_dir = os.path.join(new_root, split, "labels_from_Bbox")
    os.makedirs(new_labels_dir, exist_ok=True)

    orig_labels_dir = os.path.join(B_root, split, "labels")

    # Copy labels corresponding to images that exist in new_images_dir
    for fname in os.listdir(new_images_dir):
        if not fname.lower().endswith(".png"):
            continue

        # Possible label files: TXT or XML (keep same extension as original)
        base_name = os.path.splitext(fname)[0]

        # Check for .png label
        png_label = os.path.join(orig_labels_dir, base_name + ".png")
        if os.path.exists(png_label):
            shutil.copy2(png_label, os.path.join(new_labels_dir, base_name + ".png"))
            continue

        # Check for .tif label
        tif_label = os.path.join(orig_labels_dir, base_name + ".tif")
        if os.path.exists(tif_label):
            shutil.copy2(tif_label, os.path.join(new_labels_dir, base_name + ".tif"))
            continue

print("Finished copying corresponding labels.")'''


# Copy images that are in both datasets, respecting SemSegm split
'''
#create new folder for images that are in both datasets:
both_root = "/zfs/ai4good/datasets/tree/TreeAI/12_RGB_both"
splits = ["train", "test", "val", "pick"]

for split in splits:
    os.makedirs(os.path.join(both_root, split, "images"), exist_ok=True)



for h in common_hashes:
    for folderA, fnameA, fullA in map_A[h]:
        # Determine SemSegm split
        if "/train/" in folderA:
            split = "train"
        elif "/val/" in folderA:
            split = "val"
        elif "/test/" in folderA:
            split = "test"
        elif "/pick/" in folderA:
            split = "pick"
        else:
            print("achtong, öbbis esch komisch^^") 
            continue

        # Output folder for this split
        out_dir = os.path.join(both_root, split, "images")

        # Copy SemSegm image
        shutil.copy2(fullA, os.path.join(out_dir, fnameA))'''

#copy labels from SemSegm to new folder:
'''
both_root = "/zfs/ai4good/datasets/tree/TreeAI/12_RGB_both"
semsegm_root = "/zfs/ai4good/datasets/tree/TreeAI/12_RGB_SemSegm_640_fL"

splits = ["train", "test", "val", "pick"]

for split in splits:
    images_dir = os.path.join(both_root, split, "images")
    labels_dir = os.path.join(both_root, split, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    for fname in os.listdir(images_dir):
        if not fname.lower().endswith(".png"):
            continue

        # SemSegm label path
        semsegm_label = os.path.join(semsegm_root, split, "labels", fname)
        if os.path.exists(semsegm_label):
            shutil.copy2(semsegm_label, os.path.join(labels_dir, fname))
        else:
            print(f"Label not found for {fname} in split {split}")'''


#add labels from ObjDet (generated by SAM) to both folder
'''
both_root = "/zfs/ai4good/datasets/tree/TreeAI/12_RGB_both"
B_root = "/zfs/ai4good/student/shursc/treeAI/12_RGB_ObjDet_640_fL"

splits = ["train", "test", "val", "pick"]    

for h in common_hashes:
    if h not in map_B or h not in map_A:
        print("komisch")
        continue

    # SemSegm images for this hash
    semsegm_entries = map_A[h]  # list of (folderA, fnameA, fullA)

    # ObjDet images for this hash
    objdet_entries = map_B[h]  # list of (folderB, fnameB, fullB) len 1


    #find label in ObjDet folder: ---
    for folderB, fnameB, fullB in objdet_entries: #should only be one?
        if "/train/" in folderB:
            B_split = "train"
        elif "/val/" in folderB:
            B_split = "val"
        elif "/test/" in folderB:
            B_split = "test"
        else:
            print(f"Unknown split for {folderB}")
            continue

        label_dir = os.path.join(B_root, B_split, "labels")
        baseB, _ = os.path.splitext(fnameB)

        # ObjDet labels are .png or .tif
        possible_labels = [os.path.join(label_dir, baseB + ext) for ext in [".png", ".tif"]]
        label_path = next((p for p in possible_labels if os.path.exists(p)), None)

        if label_path is None:
            print(f"No label found for ObjDet image {fnameB}")
            continue

    #--- find corresponding directory in A
        # Copy label for each SemSegm image corresponding to this hash
        for folderA, fnameA, fullA in semsegm_entries: #for pick folder, they can contain multiple.
            # Determine split of SemSegm image
            if "/train/" in folderA:
                semsegm_split = "train"
            elif "/val/" in folderA:
                semsegm_split = "val"
            elif "/test/" in folderA:
                semsegm_split = "test"
            elif "/pick/" in folderA: 
                semsegm_split = "pick"
            else:
                print(f"Unknown split for SemSegm image {folderA}")
                continue

            # Output folder
            out_dir = os.path.join(both_root, semsegm_split, "labels_from_Bbox")
            os.makedirs(out_dir, exist_ok=True)

            # Output path uses SemSegm image filename
            out_path = os.path.join(out_dir, fnameA)
            shutil.copy2(label_path, out_path)'''


#add Bboxes to new datafile
'''
both_root = "/zfs/ai4good/datasets/tree/TreeAI/12_RGB_both"
B_root    = "/zfs/ai4good/student/shursc/treeAI/12_RGB_ObjDet_640_fL"

splits = ["train", "test", "val", "pick"]

# Ensure output dirs exist
for split in splits:
    os.makedirs(os.path.join(both_root, split, "Bboxes"), exist_ok=True)

for h in common_hashes:
    # ObjDet entry (should be exactly one per hash)
    folderB, fnameB, fullB = map_B[h][0]
    baseB, _ = os.path.splitext(fnameB)

    # Determine split of ObjDet file
    if "/train/" in folderB:
        B_split = "train"
    elif "/val/" in folderB:
        B_split = "val"
    elif "/test/" in folderB:
        B_split = "test"
    else:
        print(f"Unknown ObjDet split: {folderB}")
        continue

    # Try TXT or XML
    label_dir = os.path.join(B_root, B_split, "bboxlabels")
    possible = [os.path.join(label_dir, baseB + ".txt"),
                os.path.join(label_dir, baseB + ".xml")]

    bbox_label_path = next((p for p in possible if os.path.exists(p)), None)
    if bbox_label_path is None:
        print(f"No Bbox label found for ObjDet image {fnameB}")
        continue

    # Copy label for all SemSegm images with same hash
    for folderA, fnameA, fullA in map_A[h]:

        # Determine SemSegm split
        if "/train/" in folderA:
            A_split = "train"
        elif "/val/" in folderA:
            A_split = "val"
        elif "/test/" in folderA:
            A_split = "test"
        elif "/pick/" in folderA:
            A_split = "pick"
        else:
            print(f"Unknown SemSegm split: {folderA}")
            continue

        # Output folder
        out_dir = os.path.join(both_root, A_split, "Bboxes")
        os.makedirs(out_dir, exist_ok=True)

        # Rename to SemSegm filename
        semsegm_base, _ = os.path.splitext(fnameA)
        ext = os.path.splitext(bbox_label_path)[1]   # keep .txt or .xml

        out_path = os.path.join(out_dir, semsegm_base + ext)
        shutil.copy2(bbox_label_path, out_path)'''



#add Bbox data to _new_masks
'''
new_root = "/zfs/ai4good/datasets/tree/TreeAI/12_RGB_ObjDet_new_masks"
old_root = "/zfs/ai4good/datasets/tree/TreeAI/12_RGB_ObjDet_640_fL"   # original ObjDet folder

splits = ["train", "test", "val"]

for split in splits:
    new_img_dir = os.path.join(new_root, split, "images")
    old_bbox_dir = os.path.join(old_root, split, "labels")
    out_bbox_dir = os.path.join(new_root, split, "original_Bbox")

    os.makedirs(out_bbox_dir, exist_ok=True)

    for fname in os.listdir(new_img_dir):
        if not fname.lower().endswith(".png"):
            continue

        base = os.path.splitext(fname)[0]

        # Possible Bbox label formats
        candidates = [
            os.path.join(old_bbox_dir, base + ".txt"),
            os.path.join(old_bbox_dir, base + ".xml")
        ]

        src = next((p for p in candidates if os.path.exists(p)), None)

        if src is None:
            print(f"WARNING: No Bbox found for {split}/{fname}")
            continue

        shutil.copy2(src, os.path.join(out_bbox_dir, os.path.basename(src)))'''

#add bboxes with changed labels
'''
new_root = "/zfs/ai4good/datasets/tree/TreeAI/12_RGB_ObjDet_new_masks"
old_root = "/zfs/ai4good/datasets/tree/TreeAI/12_RGB_ObjDet_640_fL"
mapping_json = "label_mapping.json"


# Load mapping: old → new class id
with open(mapping_json, "r") as f:
    mapping = json.load(f)["dataset"]
mapping = {int(k): int(v) for k, v in mapping.items()}

splits = ["train", "test", "val"]


def remap_txt(src_path, dst_path):
    """YOLO .txt format:  class x_center y_center w h"""
    out_lines = []
    with open(src_path, "r") as f:
        for line in f:
            if line.strip() == "":
                continue
            parts = line.strip().split()
            old = int(float(parts[0]))
            new = mapping.get(old, old)   # if missing, keep same
            parts[0] = str(new)
            out_lines.append(" ".join(parts))

    with open(dst_path, "w") as f:
        for l in out_lines:
            f.write(l + "\n")


def remap_xml(src_path, dst_path):
    """Pascal VOC/ArcGIS .xml format: <object><name>ID</name></object>"""
    tree = ET.parse(src_path)
    root = tree.getroot()

    for obj in root.findall("object"):
        name_tag = obj.find("name")
        if name_tag is not None:
            old = int(name_tag.text)
            new = mapping.get(old, old)
            name_tag.text = str(new)

    tree.write(dst_path)


for split in splits:
    new_img_dir = os.path.join(new_root, split, "images")
    old_bbox_dir = os.path.join(old_root, split, "labels")
    new_bbox_dir = os.path.join(new_root, split, "new_Bbox")

    os.makedirs(new_bbox_dir, exist_ok=True)

    for fname in os.listdir(new_img_dir):
        if not fname.lower().endswith(".png"):
            continue

        base = os.path.splitext(fname)[0]

        candidates = [
            os.path.join(old_bbox_dir, base + ".txt"),
            os.path.join(old_bbox_dir, base + ".xml"),
        ]

        src = next((p for p in candidates if os.path.exists(p)), None)

        if src is None:
            print(f"WARNING: No Bbox found for {split}/{fname}")
            continue

        ext = os.path.splitext(src)[1]
        dst = os.path.join(new_bbox_dir, base + ext)

        # Remap according to type
        if ext == ".txt":
            remap_txt(src, dst)
        elif ext == ".xml":
            remap_xml(src, dst)
        else:
            print(f"Unknown Bbox label format: {src}")'''

            