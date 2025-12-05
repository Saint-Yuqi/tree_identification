#!/usr/bin/env python3
"""
Overlay semantic segmentation label masks onto images for visualization.

Assumptions:
- Images are in a directory like .../test/images
- Labels are in a directory like .../test/labels
- Filenames match between images and labels (e.g., 000000001745.png)
- Label PNGs contain integer class ids (single-channel). If labels are RGB,
  only one channel will be used.

Usage: (change directoriepath)
  python3 overlay_semseg_masks.py \
    --images_dir /home/c/shursc/data/TreeAI/12_RGB_SemSegm_640_fL/train/images \
    --labels_dir /home/c/shursc/data/TreeAI/12_RGB_SemSegm_640_fL/train/labels \
    --output_dir /home/c/shursc/data/TreeAI/12_RGB_SemSegm_640_fL/train/overlays \
    --alpha 0.5 \
    --legend \
    --classes_yaml /home/c/shursc/code/tree_identification/configs/data/treeAI_classes.yaml
"""

import argparse
import os
import sys
from typing import Dict, Tuple, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def build_default_palette(num_classes: int = 256) -> np.ndarray:
    """Create a visually distinct color palette for up to num_classes classes.

    Returns an array of shape (num_classes, 3) with uint8 RGB colors.
    """
    # Use a simple, deterministic palette based on bit patterns (like PASCAL VOC)
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for j in range(num_classes):
        lab = j
        r = g = b = 0
        i = 0
        while lab:
            r |= (((lab >> 0) & 1) << (7 - i))
            g |= (((lab >> 1) & 1) << (7 - i))
            b |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
        palette[j] = np.array([r, g, b], dtype=np.uint8)
    # Make background (0) semi-neutral if desired; keep as is for determinism
    return palette


def load_classes_yaml(path: str) -> Dict[int, Dict[str, str]]:
    """Load classes mapping from a YAML file with structure:
    classes:
      0: { name: "Background", color: "#000000" }
      1: { name: "class1", color: "#ff0000" }
    Returns dict mapping int id -> {name: str, color: str(hex)}
    """
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    classes = data.get('classes', {})
    # Normalize keys to int
    result = {}
    for k, v in classes.items():
        try:
            kid = int(k)
        except Exception:
            continue
        result[kid] = {
            'name': v.get('name', str(kid)),
            'color': v.get('color', '#000000')
        }
    return result


def hex_to_rgb(hex_color: str) -> np.ndarray:
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])
    return np.array(tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4)), dtype=np.uint8)


def build_palette_from_classes(classes: Dict[int, Dict[str, str]], max_classes: int) -> np.ndarray:
    palette = build_default_palette(max_classes)
    for cid, meta in classes.items():
        if 0 <= cid < max_classes:
            palette[cid] = hex_to_rgb(meta.get('color', '#000000'))
    return palette


def _measure_text(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, text: str) -> Tuple[int, int]:
    """Return width, height of text using robust Pillow APIs across versions."""
    # Prefer font.getbbox when available
    if hasattr(font, 'getbbox'):
        l, t, r, b = font.getbbox(text)
        return r - l, b - t
    # Fallback to draw.textbbox
    if hasattr(draw, 'textbbox'):
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return r - l, b - t
    # Final fallback to deprecated font.getsize if present
    if hasattr(font, 'getsize'):
        w, h = font.getsize(text)
        return int(w), int(h)
    # Worst-case constant
    return max(6, len(text) * 6), 12


def render_legend(base_img: Image.Image, present_ids: List[int], palette: np.ndarray, id_to_name: Dict[int, str]) -> Image.Image:
    """Render a legend below the image showing color squares and label names for present_ids.
    Returns a new PIL Image with the legend appended at the bottom.
    """
    # Sort ids, keep -1 at end if present
    ids = [i for i in sorted([i for i in present_ids if i != -1])] + ([ -1 ] if (-1 in present_ids) else [])
    if not ids:
        return base_img

    # Legend layout
    padding = 10
    swatch_size = 18
    gap = 8
    font = ImageFont.load_default()

    # Compute legend width needed
    draw_tmp = ImageDraw.Draw(base_img)
    entries = []
    for cid in ids:
        name = id_to_name.get(cid, str(cid))
        text_w, text_h = _measure_text(draw_tmp, font, name)
        entries.append((cid, name, text_w, text_h))
    # Arrange entries in columns to fit width
    img_w, img_h = base_img.size
    col_w = max((swatch_size + gap + e[2]) for e in entries) + padding
    cols = max(1, img_w // col_w)
    rows = (len(entries) + cols - 1) // cols
    legend_h = padding*2 + rows * (max(swatch_size, entries[0][3]) + gap) - gap

    # Create new image with space for legend
    out = Image.new('RGB', (img_w, img_h + legend_h), color=(255, 255, 255))
    out.paste(base_img, (0, 0))
    draw = ImageDraw.Draw(out)

    # Draw entries
    y0 = img_h + padding
    for idx, (cid, name, text_w, text_h) in enumerate(entries):
        r = idx // cols
        c = idx % cols
        x = padding + c * col_w
        y = y0 + r * (max(swatch_size, text_h) + gap)
        color = tuple(int(x) for x in palette[cid].tolist()) if cid != -1 else tuple(int(x) for x in palette[-1].tolist())
        draw.rectangle([x, y, x+swatch_size, y+swatch_size], fill=color, outline=(0,0,0))
        draw.text((x+swatch_size+gap, y + (swatch_size - text_h)//2), name, fill=(0,0,0), font=font)
    return out


def colorize_label(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Map integer label mask to an RGB image using the given palette.

    mask: HxW int array (any integer dtype)
    palette: (K,3) uint8 array; if mask contains >=K values they will wrap
    """
    if mask.ndim != 2:
        raise ValueError("Label mask must be 2D (HxW)")
    # Map indices to palette; ensure -1 maps to the last color distinctly
    palette_size = palette.shape[0]
    indices = mask.astype(np.int64)
    neg1 = (indices == -1)
    # Wrap all indices into range [0, palette_size)
    indices = indices % palette_size
    if np.any(neg1):
        indices[neg1] = palette_size - 1
    colored = palette[indices]
    return colored.astype(np.uint8)  # HxW x3


def overlay(image_rgb: np.ndarray, mask_rgb: np.ndarray, alpha: float) -> np.ndarray:
    """Alpha-blend mask_rgb over image_rgb.

    image_rgb: HxW x3 uint8
    mask_rgb: HxW x3 uint8
    alpha: in [0,1], higher shows mask more strongly
    """
    if image_rgb.shape != mask_rgb.shape:
        raise ValueError("Image and mask must have the same shape for overlay")
    img = image_rgb.astype(np.float32)
    msk = mask_rgb.astype(np.float32)
    blended = (1.0 - alpha) * img + alpha * msk
    return np.clip(blended, 0, 255).astype(np.uint8)


def load_image_rgb(path: str) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))


def load_label_mask(path: str) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.array(im)
        if arr.ndim == 3:
            # If label is RGB, take first channel (common in some exports)
            arr = arr[..., 0]
        return arr


def find_pairs(images_dir: str, labels_dir: str) -> Dict[str, Tuple[str, str]]:
    """Match image and label files by filename (stem + extension equal)."""
    images = {f: os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))}
    labels = {f: os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))}
    common = sorted(set(images.keys()) & set(labels.keys()))
    return {fname: (images[fname], labels[fname]) for fname in common}


def main() -> int:
    parser = argparse.ArgumentParser(description="Overlay segmentation labels onto images")
    parser.add_argument("--images_dir", required=True, type=str, help="Path to directory with images")
    parser.add_argument("--labels_dir", required=True, type=str, help="Path to directory with label masks")
    parser.add_argument("--output_dir", required=True, type=str, help="Path to write overlay images")
    parser.add_argument("--alpha", default=0.5, type=float, help="Mask overlay alpha in [0,1]")
    parser.add_argument("--max_classes", default=256, type=int, help="Max classes supported in palette")
    parser.add_argument("--color_label0", type=str, default=None, help="Override RGB for label 0, e.g. '255,255,255'")
    parser.add_argument("--color_label_neg1", type=str, default=None, help="Override RGB for label -1, e.g. '255,255,0'")
    parser.add_argument("--print_labels", action="store_true", help="Print unique label IDs per image with counts and colors")
    parser.add_argument("--save_mask_only", action="store_true", help="Also save the standalone colorized mask image for debugging")
    parser.add_argument("--classes_yaml", type=str, default=None, help="Optional path to classes YAML with names and hex colors")
    parser.add_argument("--legend", action="store_true", help="Append a legend with label-color mapping to each output image")
    args = parser.parse_args()

    if not (0.0 <= args.alpha <= 1.0):
        print("--alpha must be in [0,1]", file=sys.stderr)
        return 2

    # Normalize and expand paths
    args.images_dir = os.path.abspath(os.path.expanduser(args.images_dir))
    args.labels_dir = os.path.abspath(os.path.expanduser(args.labels_dir))
    args.output_dir = os.path.abspath(os.path.expanduser(args.output_dir))

    if not os.path.isdir(args.images_dir):
        print(f"Images dir not found: {args.images_dir}", file=sys.stderr)
        return 2
    if not os.path.isdir(args.labels_dir):
        print(f"Labels dir not found: {args.labels_dir}", file=sys.stderr)
        return 2

    # Check writability of output directory (or its parent if it does not exist yet)
    out_parent = args.output_dir if os.path.isdir(args.output_dir) else os.path.dirname(args.output_dir) or "/"
    if not os.access(out_parent, os.W_OK):
        print(f"Output path not writable: {args.output_dir}. Choose a directory under your home or a writable location.", file=sys.stderr)
        return 2

    ensure_dir(args.output_dir)

    pairs = find_pairs(args.images_dir, args.labels_dir)
    if not pairs:
        print("No matching image/label pairs found.", file=sys.stderr)
        return 1

    id_to_name = {}
    if args.classes_yaml is not None:
        classes = load_classes_yaml(args.classes_yaml)
        palette = build_palette_from_classes(classes, args.max_classes)
        id_to_name = {k: v['name'] for k, v in classes.items()}
    else:
        palette = build_default_palette(args.max_classes)

    def parse_rgb(spec: str) -> np.ndarray:
        parts = [p.strip() for p in spec.split(',')]
        if len(parts) != 3:
            raise ValueError("RGB must be 'R,G,B'")
        vals = [int(p) for p in parts]
        if any(v < 0 or v > 255 for v in vals):
            raise ValueError("RGB values must be in [0,255]")
        return np.array(vals, dtype=np.uint8)

    # Optional explicit colors for label 0 and -1 (mapped to last index)
    if args.color_label0 is not None:
        palette[0] = parse_rgb(args.color_label0)
    if args.color_label_neg1 is not None:
        palette[-1] = parse_rgb(args.color_label_neg1)

    # Echo applied colors for clarity when overrides are requested
    if args.color_label0 is not None or args.color_label_neg1 is not None:
        msg = {
            "label_0": palette[0].tolist(),
            "label_-1": palette[-1].tolist(),
        }
        print(f"Applied palette overrides: {msg}")

    processed = 0
    skipped = 0
    for fname, (img_path, label_path) in pairs.items():
        try:
            img = load_image_rgb(img_path)
            label = load_label_mask(label_path)
            # Map index 65535 to 0
            label[label == 65535] = 0
            ids, counts = np.unique(label, return_counts=True)
            if args.print_labels:
                # Map IDs to colors as used after potential resize (same logic applies)
                palette_size = palette.shape[0]
                mapped_indices = ids.astype(np.int64) % palette_size
                # Ensure -1 maps to last color
                mapped_indices[ids == -1] = palette_size - 1
                colors = [palette[i].tolist() for i in mapped_indices]
                summary = {int(i): {"count": int(c), "color": col, "name": id_to_name.get(int(i), str(int(i)))} for i, c, col in zip(ids.tolist(), counts.tolist(), colors)}
                print(f"{fname}: {summary}")
            if label.shape[:2] != img.shape[:2]:
                # Resize label to image size using nearest to preserve class ids
                label_img = Image.fromarray(label)
                label_img = label_img.resize((img.shape[1], img.shape[0]), resample=Image.NEAREST)
                label = np.array(label_img)

            mask_rgb = colorize_label(label, palette)

            # Optionally save the colorized mask alone for verification
            if args.save_mask_only:
                mask_only_dir = os.path.join(args.output_dir, "masks")
                ensure_dir(mask_only_dir)
                Image.fromarray(mask_rgb).save(os.path.join(mask_only_dir, fname))

            out = overlay(img, mask_rgb, args.alpha)

            if args.legend:
                # Build id->name mapping; default to string of id
                id_name_map = {int(i): id_to_name.get(int(i), str(int(i))) for i in ids.tolist()}
                out_pil = Image.fromarray(out)
                out_pil = render_legend(out_pil, list(id_name_map.keys()), palette, id_name_map)
                out = np.array(out_pil)

            out_path = os.path.join(args.output_dir, fname)
            Image.fromarray(out).save(out_path)
            processed += 1
        except Exception as e:
            print(f"[WARN] Skipping {fname}: {e}", file=sys.stderr)
            skipped += 1

    print(f"Processed: {processed}, Skipped: {skipped}, Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


