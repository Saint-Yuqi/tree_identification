import matplotlib.pyplot as plt
import os
from PIL import Image
import yaml
import numpy as np

'''
create figures: real image with bounding boxes vs. "real" masks vs. generated masks
3 images in one row
one row per image
input the image paths in ~line 120
'''


def load_image_rgb(path):
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))

def load_label_mask(path):
    with Image.open(path) as im:
        arr = np.array(im)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        return arr

def overlay(image_rgb, mask_rgb, alpha=0.5):
    img = image_rgb.astype(np.float32)
    msk = mask_rgb.astype(np.float32)
    blended = (1 - alpha) * img + alpha * msk
    return np.clip(blended, 0, 255).astype(np.uint8)

def colorize_label(mask, palette):
    H, W = mask.shape
    palette_size = palette.shape[0]
    idx = mask % palette_size
    idx[mask == -1] = palette_size - 1
    return palette[idx]


# Load class names + colors from YAML
def load_classes_yaml(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)["classes"]
    id_to_name = {int(k): v["name"] for k, v in data.items()}
    id_to_color = {int(k): v["color"] for k, v in data.items()}
    return id_to_name, id_to_color

def load_bbox_yolo(path):
    """
    Loads YOLO-format bounding boxes
    Returns a list of (class_id, x_center, y_center, w, h)
    Normalised coordinates ∈ [0,1].
    """
    boxes = []
    with open(path, "r") as f:
        for line in f:
            if line.strip() == "":
                continue
            cls, xc, yc, w, h = map(float, line.strip().split())
            boxes.append((int(cls), xc, yc, w, h))
    return boxes


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return np.array([int(hex_color[i:i+2], 16) for i in (0, 2, 4)], dtype=np.uint8)


# Build palette
def build_palette(id_to_color):
    max_id = max(id_to_color.keys())
    palette = np.zeros((max_id + 2, 3), dtype=np.uint8)  
    for cid, hex_col in id_to_color.items():
        palette[cid] = hex_to_rgb(hex_col)
    palette[-1] = np.array([255, 255, 0], dtype=np.uint8)  # for -1 if present
    return palette


def draw_bboxes(image_rgb, boxes, palette):
    """
    Draws coloured rectangles over an image.
    Each class ID is drawn using palette[class_id].
    """
    img = image_rgb.copy()
    H, W, _ = img.shape

    for cls, xc, yc, w, h in boxes:
        # convert normalised → pixel coordinates
        bw = int(w * W)
        bh = int(h * H)
        cx = int(xc * W)
        cy = int(yc * H)

        x1 = max(0, cx - bw // 2)
        y1 = max(0, cy - bh // 2)
        x2 = min(W - 1, cx + bw // 2)
        y2 = min(H - 1, cy + bh // 2)

        color = tuple(int(c) for c in palette[cls])  # RGB

        # draw rectangle border
        img[y1:y1+3, x1:x2] = color
        img[y2-3:y2, x1:x2] = color
        img[y1:y2, x1:x1+3] = color
        img[y1:y2, x2-3:x2] = color

    return img


# ----------------------------------------------------------------------
# Main plotting code
# ----------------------------------------------------------------------

classes_yaml = "/home/c/shursc/code/tree_identification/configs/data/treeAI_classes.yaml"   
id_to_name, id_to_color = load_classes_yaml(classes_yaml)
palette = build_palette(id_to_color)


#folders
images= "/zfs/ai4good/datasets/tree/TreeAI/12_RGB_both/train/images"
labels_real = "/zfs/ai4good/datasets/tree/TreeAI/12_RGB_both/train/labels"
labels_from_Bbox = "/zfs/ai4good/datasets/tree/TreeAI/12_RGB_both/train/labels_Bbox_changed"
labels_Bbox_changed = "/zfs/ai4good/datasets/tree/TreeAI/12_RGB_both/train/new_Bbox"

image_list = ["000000001508", "000000001856", "000000001532","000000000669"]

#what we used in the paper:
#main_text: ["000000000327", "000000001302"]
#appemndix_1: ["000000001134", "000000001367", "000000001240", "000000001353"]
#appendix_2: ["000000001508", "000000001856", "000000001532","000000000669"]
#for one page image good sizes are as follows: 4 images,figsize(14, 4*len(image_list))), 
# set_title fontsize 16 or 17, fontsize = 15



# titles per column
col_titles = ["Bounding boxes", "Real masks", "Generated masks"]

fig, axes = plt.subplots(
    nrows=len(image_list),
    ncols=3,
    figsize=(15, 4 * len(image_list))
)

if len(image_list) == 1:
    axes = np.expand_dims(axes, 0)

for row_idx, image in enumerate(image_list):

    # paths
    img_path = os.path.join(images, image + ".png")
    mask_real_path = os.path.join(labels_real, image + ".png")
    mask_from_bbox_path = os.path.join(labels_from_Bbox, image + ".png")
    bbox_path = os.path.join(labels_Bbox_changed, image + ".txt")

    # load data
    img = load_image_rgb(img_path)
    m_real = load_label_mask(mask_real_path)
    m_from_bbox = load_label_mask(mask_from_bbox_path)
    bbox = load_bbox_yolo(bbox_path)

    # colorize
    c_real = colorize_label(m_real, palette)
    c_from_bbox = colorize_label(m_from_bbox, palette)

    # overlays
    ov_bbox = draw_bboxes(img, bbox, palette)
    ov_real = overlay(img, c_real)
    ov_from_bbox = overlay(img, c_from_bbox)

    images_to_plot = [ov_bbox, ov_real, ov_from_bbox]
    masks_used = [bbox, m_real, m_from_bbox]

    # plot row
    for col_idx in range(3):

        ax = axes[row_idx, col_idx]
        ax.imshow(images_to_plot[col_idx])
        ax.axis("off")

        if row_idx == 0:
            ax.set_title(col_titles[col_idx], fontsize=16, pad=10)

    # Build legend for this row
    bbox_classes = sorted(set(cls for cls,_,_,_,_ in bbox))
    uniques = sorted(set(np.unique(np.concatenate([m_real, m_from_bbox]))).union(bbox_classes))


    legend_patches = []
    for uid in uniques:
        rgb = palette[uid] / 255
        lbl = id_to_name.get(uid, str(uid))
        patch = plt.Line2D([0], [0], marker='s', color=rgb, markersize=12, linestyle='None', label=lbl)
        legend_patches.append(patch)

    axes[row_idx, 2].legend(
        handles=legend_patches,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.,
        #title="Classes",
        fontsize=15,
        frameon=False   )

plt.tight_layout()
plt.savefig("resulting_images/real_vs_created_vs_bbox_app2.pdf")