from PIL import Image
import numpy as np
import cv2
import csv
import os
import xml.etree.ElementTree as ET
from collections import Counter


"""
this file was used to determine which labels can be correct. It does the following

- show image with Bounding Boxes
    -input: Image & label paths 
            -labels which should be shown (numbers)
    -output: image with drawn bboxes for the specified classes
- count number of bounding bboxes per class (~line 80)
    -output: in terminal bounding boxes per class, and as a csv file
"""

# Load image
image_link = "/home/c/shursc/data/TreeAI/12_RGB_SemSegm_640_fL/train/labels/000000002490.png"
img = Image.open(image_link)

# Convert to numpy array (keeps integer values)
mask = np.array(img)

print(mask.dtype)      # usually uint8 for labels 0–255
print(mask.shape)      # height × width
print(np.unique(mask)) # 


# show image with bounding boxes
#------------------------------------------------------------------
number = "0913"
image_path = f"/home/c/shursc/data/TreeAI/12_RGB_both/train/images/00000000{number}.png"
label_path = f"/home/c/shursc/data/TreeAI/12_RGB_both/train/bboxes/00000000{number}.txt"
output_folder = "checkstuff/tanne"
keep_labels = {40, 47, 52}   # nur diese Bboxes anzeigen
# ------------------------------------------------------------------
os.makedirs(output_folder, exist_ok=True)


img = cv2.imread(image_path)
img_h, img_w = img.shape[:2]



with open(label_path, "r") as f:
    lines = f.readlines()

#draw boundingboxes
for line in lines:
    cls, x_c, y_c, w, h = map(float, line.split())
    cls = int(cls)

    if cls not in keep_labels:
        continue

    # YOLO -> pixel coordinates
    x_c *= img_w
    y_c *= img_h
    w *= img_w
    h *= img_h

    x1 = int(x_c - w/2)
    y1 = int(y_c - h/2)
    x2 = int(x_c + w/2)
    y2 = int(y_c + h/2)

    
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, str(cls), (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

output_path = os.path.join(output_folder, os.path.basename(image_path))
cv2.imwrite(output_path, img)
print(f"Saved result to {output_path}")




# --------------------------------------------------------------------
# Count number of Bboxes per species/class in new_Bbox/test
# --------------------------------------------------------------------

new_root = "/zfs/ai4good/datasets/tree/TreeAI/12_RGB_ObjDet_new_masks"
bbox_dir = os.path.join(new_root, "test", "new_Bbox")


bbox_counter = Counter()

def count_from_txt(path):
    """YOLO txt: class cx cy w h"""
    with open(path, "r") as f:
        for line in f:
            if line.strip() == "":
                continue
            cls_id = int(float(line.split()[0]))
            bbox_counter[cls_id] += 1

def count_from_xml(path):
    """Pascal VOC XML: <object><name>cls</name></object>"""
    tree = ET.parse(path)
    root = tree.getroot()
    for obj in root.findall("object"):
        name_tag = obj.find("name")
        if name_tag is not None:
            cls_id = int(name_tag.text)
            bbox_counter[cls_id] += 1

for fname in os.listdir(bbox_dir):
    if not (fname.endswith(".txt") or fname.endswith(".xml")):
        continue

    full = os.path.join(bbox_dir, fname)

    if fname.endswith(".txt"):
        count_from_txt(full)
    else:
        count_from_xml(full)


print("Bounding boxes per species in test/new_Bbox:")
for cls_id, count in sorted(bbox_counter.items()):
    print(f"Class {cls_id}: {count}")


#save output to csv
csv_path = os.path.join(bbox_dir, "bbox_counts_additional_12_images.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["class_id", "count"])
    for cls_id, count in sorted(bbox_counter.items()):
        writer.writerow([cls_id, count])

print(f"Saved CSV to {csv_path}")

