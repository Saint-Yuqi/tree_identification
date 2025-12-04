import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT_NODE = "root"
LEVELS = ("class", "order", "family", "genus")
BASE_DIR = os.path.dirname(__file__)

ORDER_IGNORE_INDICES = {37, 60}
GENUS_IGNORE_INDICES = {5, 11, 43, 50, 56, 58, 59, 61}
IGNORE_INDICES = ORDER_IGNORE_INDICES | GENUS_IGNORE_INDICES

TAXONOMY_ROWS: Sequence[Sequence[str]] = [
    ("Background", "", "", "", ""),
    ("betula_papyrifera", "Magnoliopsida", "Fagales", "Betulaceae", "Betula"),
    ("tsuga_canadensis", "Pinopsida", "Pinales", "Pinaceae", "Tsuga"),
    ("picea_abies", "Pinopsida", "Pinales", "Pinaceae", "Picea"),
    ("acer_saccharum", "Magnoliopsida", "Sapindales", "Sapindaceae", "Acer"),
    ("betula_sp.", "", "", "", "Betula"),
    ("pinus_sylvestris", "Pinopsida", "Pinales", "Pinaceae", "Pinus"),
    ("picea_rubens", "Pinopsida", "Pinales", "Pinaceae", "Picea"),
    ("betula_alleghaniensis", "Magnoliopsida", "Fagales", "Betulaceae", "Betula"),
    ("larix_decidua", "Pinopsida", "Pinales", "Pinaceae", "Larix"),
    ("fagus_grandifolia", "Magnoliopsida", "Fagales", "Fagaceae", "Fagus"),
    ("picea_sp.", "", "", "", "Picea"),
    ("fagus_sylvatica", "Magnoliopsida", "Fagales", "Fagaceae", "Fagus"),
    ("dead_tree", "", "", "", ""),
    ("acer_pensylvanicum", "Magnoliopsida", "Sapindales", "Sapindaceae", "Acer"),
    ("populus_balsamifera", "Magnoliopsida", "Malpighiales", "Salicaceae", "Populus"),
    ("quercus_ilex", "Magnoliopsida", "Fagales", "Fagaceae", "Quercus"),
    ("quercus_robur", "Magnoliopsida", "Fagales", "Fagaceae", "Quercus"),
    ("pinus_strobus", "Pinopsida", "Pinales", "Pinaceae", "Pinus"),
    ("larix_laricina", "Pinopsida", "Pinales", "Pinaceae", "Larix"),
    ("larix_gmelinii", "Pinopsida", "Pinales", "Pinaceae", "Larix"),
    ("pinus_pinea", "Pinopsida", "Pinales", "Pinaceae", "Pinus"),
    ("populus_grandidentata", "Magnoliopsida", "Malpighiales", "Salicaceae", "Populus"),
    ("pinus_montezumae", "Pinopsida", "Pinales", "Pinaceae", "Pinus"),
    ("abies_alba", "Pinopsida", "Pinales", "Pinaceae", "Abies"),
    ("betula_pendula", "Magnoliopsida", "Fagales", "Betulaceae", "Betula"),
    ("pseudotsuga_menziesii", "Pinopsida", "Pinales", "Pinaceae", "Pseudotsuga"),
    ("fraxinus_nigra", "Magnoliopsida", "Lamiales", "Oleaceae", "Fraxinus"),
    ("dacrydium_cupressinum", "Pinopsida", "Pinales", "Podocarpaceae", "Dacrydium"),
    ("cedrus_libani", "Pinopsida", "Pinales", "Pinaceae", "Cedrus"),
    ("acer_pseudoplatanus", "Magnoliopsida", "Sapindales", "Sapindaceae", "Acer"),
    ("pinus_elliottii", "Pinopsida", "Pinales", "Pinaceae", "Pinus"),
    ("cryptomeria_japonica", "Pinopsida", "Pinales", "Cupressaceae", "Cryptomeria"),
    ("pinus_koraiensis", "Pinopsida", "Pinales", "Pinaceae", "Pinus"),
    ("abies_holophylla", "Pinopsida", "Pinales", "Pinaceae", "Abies"),
    ("alnus_glutinosa", "Magnoliopsida", "Fagales", "Betulaceae", "Alnus"),
    ("fraxinus_excelsior", "Magnoliopsida", "Lamiales", "Oleaceae", "Fraxinus"),
    ("coniferous", "", "", "", ""),
    ("eucalyptus_globulus", "Magnoliopsida", "Myrtales", "Myrtaceae", "Eucalyptus"),
    ("pinus_nigra", "Pinopsida", "Pinales", "Pinaceae", "Pinus"),
    ("quercus_rubra", "Magnoliopsida", "Fagales", "Fagaceae", "Quercus"),
    ("tilia_europaea", "Magnoliopsida", "Malvales", "Malvaceae", "Tilia"),
    ("abies_firma", "Pinopsida", "Pinales", "Pinaceae", "Abies"),
    ("acer_sp.", "", "", "", "Acer"),
    ("metrosideros_umbellata", "Magnoliopsida", "Myrtales", "Myrtaceae", "Metrosideros"),
    ("acer_rubrum", "Magnoliopsida", "Sapindales", "Sapindaceae", "Acer"),
    ("picea_mariana", "Pinopsida", "Pinales", "Pinaceae", "Picea"),
    ("abies_balsamea", "Pinopsida", "Pinales", "Pinaceae", "Abies"),
    ("castanea_sativa", "Magnoliopsida", "Fagales", "Fagaceae", "Castanea"),
    ("tilia_cordata", "Magnoliopsida", "Malvales", "Malvaceae", "Tilia"),
    ("populus_sp.", "", "", "", "Populus"),
    ("crataegus_monogyna", "Magnoliopsida", "Rosales", "Rosaceae", "Crataegus"),
    ("quercus_petraea", "Magnoliopsida", "Fagales", "Fagaceae", "Quercus"),
    ("acer_platanoides", "Magnoliopsida", "Sapindales", "Sapindaceae", "Acer"),
    ("robinia_pseudoacacia", "Magnoliopsida", "Fabales", "Fabaceae", "Robinia"),
    ("fagus_crenata", "Magnoliopsida", "Fagales", "Fagaceae", "Fagus"),
    ("quercus_sp.", "", "", "", "Quercus"),
    ("salix_alba", "Magnoliopsida", "Malpighiales", "Salicaceae", "Salix"),
    ("pinus_sp.", "", "", "", "Pinus"),
    ("carpinus_sp.", "", "", "", "Carpinus"),
    ("deciduous", "", "", "", ""),
    ("salix_sp.", "", "", "", "Salix"),
]


def prepare_entries() -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    keys = ("label", *LEVELS)
    for row in TAXONOMY_ROWS:
        entry = dict(zip(keys, row))
        entry["label"] = entry["label"].strip()
        for level in LEVELS:
            entry[level] = entry[level].strip()
        entries.append(entry)
    return entries


def build_paths(entries: List[Dict[str, str]]) -> Dict[str, List[str]]:
    paths: Dict[str, List[str]] = {}
    for entry in entries:
        current_path = [ROOT_NODE]
        for level in LEVELS:
            value = entry[level]
            if value:
                current_path.append(f"{level}:{value}")
        current_path.append(f"label:{entry['label']}")
        paths[entry["label"]] = current_path
    return paths


def path_distance(path_a: List[str], path_b: List[str]) -> int:
    common = 0
    for node_a, node_b in zip(path_a, path_b):
        if node_a == node_b:
            common += 1
        else:
            break
    return (len(path_a) - common) + (len(path_b) - common)


def load_existing_labels(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    labels: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _, value = line.split(":", 1)
            labels.append(value.strip())
    return labels

def save_taxonomy_distance_matrix(entries: List[Dict[str, str]]) -> torch.Tensor:
    labels = [entry["label"] for entry in entries]
    paths = build_paths(entries)

    existing_labels_path = os.path.join(BASE_DIR, "class_labels.txt")
    existing_labels = load_existing_labels(existing_labels_path)

    if existing_labels and existing_labels != labels:
        raise ValueError(
            "Taxonomy rows do not match existing class_labels.txt order. "
            "Update TAXONOMY_ROWS or class_labels.txt so the orders align."
        )

    num_classes = len(labels)
    matrix: List[List[float]] = [[0.0 for _ in range(num_classes)] for _ in range(num_classes)]

    for i, label_i in enumerate(labels):
        path_i = paths[label_i]
        for j in range(i, num_classes):
            path_j = paths[labels[j]]
            distance = float(path_distance(path_i, path_j))
            matrix[i][j] = distance
            matrix[j][i] = distance

    hierarchical_max = max(max(row) for row in matrix)

    def force_max_distance(label: str, value: float) -> None:
        if label not in paths:
            return
        idx = labels.index(label)
        for j in range(num_classes):
            if j == idx:
                continue
            matrix[idx][j] = value
            matrix[j][idx] = value

    for special_label in ("Background", "dead_tree"):
        force_max_distance(special_label, hierarchical_max)

    tensor = torch.tensor(matrix, dtype=torch.float32)
    tensor_path = os.path.join(BASE_DIR, "taxonomy_distance_matrix.pt")
    csv_path = os.path.join(BASE_DIR, "taxonomy_distance_matrix.csv")

    torch.save(tensor, tensor_path)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + labels)
        for label, row in zip(labels, matrix):
            writer.writerow([label] + [f"{value:.2f}" for value in row])

    max_distance = max(max(row) for row in matrix)
    print(f"Saved taxonomy distance matrix ({num_classes}x{num_classes})")
    print(f" - PT:  {tensor_path}")
    print(f" - CSV: {csv_path}")
    print(f"Max shortest-path distance: {max_distance:.2f}")
    return tensor


def sort_species_by_genus(
    entries: List[Dict[str, str]],
    ignore_indices: Iterable[int] = IGNORE_INDICES,
) -> Tuple[List[int], List[str]]:
    ignore_set = set(ignore_indices)
    sortable: List[Tuple[str, str, int]] = []
    for idx, entry in enumerate(entries):
        if idx in ignore_set:
            continue
        genus = entry["genus"] or ""
        genus_key = genus.lower() if genus else "zzzz_unknown"
        sortable.append((genus_key, entry["label"].lower(), idx))
    sortable.sort()
    sorted_indices = [idx for _, _, idx in sortable]
    sorted_labels = [entries[idx]["label"] for idx in sorted_indices]
    return sorted_indices, sorted_labels


def _to_numpy(array_like) -> np.ndarray:
    if isinstance(array_like, np.ndarray):
        return array_like
    try:
        if isinstance(array_like, torch.Tensor):
            return array_like.cpu().numpy()
    except Exception:
        pass
    return np.asarray(array_like)


def load_array(path: Path):
    ext = path.suffix.lower()
    if ext == ".npy":
        return np.load(path, allow_pickle=True)
    if ext == ".npz":
        with np.load(path, allow_pickle=True) as data:
            return {k: data[k] for k in data.files}
    if ext == ".pt":
        return torch.load(path, map_location="cpu")
    if ext in (".csv", ".txt"):
        try:
            return np.loadtxt(path, delimiter=",")
        except ValueError:
            with path.open(newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = [row for row in reader if row]
            try:
                return np.asarray([[float(v) for v in row[1:]] for row in rows[1:]], dtype=float)
            except Exception as exc:
                raise ValueError(f"Could not parse numeric matrix from {path}") from exc
    raise ValueError(f"Unsupported file extension for {path}")


def _extract_preds_targets(container: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    key_map = {k.lower(): k for k in container.keys()}
    pred_key = next((key_map[k] for k in ("preds", "predictions", "outputs", "logits") if k in key_map), None)
    target_key = next((key_map[k] for k in ("targets", "labels", "gts", "gt", "masks") if k in key_map), None)
    if pred_key is None or target_key is None:
        raise ValueError("Could not find prediction/target keys in container.")
    return _to_numpy(container[pred_key]), _to_numpy(container[target_key])


def load_preds_and_targets(pred_path: Path, target_path: Optional[Path]) -> Tuple[np.ndarray, np.ndarray]:
    pred_obj = load_array(pred_path)
    if isinstance(pred_obj, dict) and target_path is None:
        preds, targets = _extract_preds_targets(pred_obj)
    elif target_path is not None:
        preds = _to_numpy(pred_obj)
        targets = _to_numpy(load_array(target_path))
    else:
        raise ValueError("Provide both prediction and target arrays or a single container file with both.")
    if preds.shape != targets.shape:
        raise ValueError(f"Shape mismatch between preds {preds.shape} and targets {targets.shape}.")
    return preds, targets


def load_confusion_matrix(path: Path) -> np.ndarray:
    conf_obj = load_array(path)
    if isinstance(conf_obj, dict):
        key_map = {k.lower(): k for k in conf_obj.keys()}
        key = next((key_map[k] for k in ("confusion", "conf", "cm") if k in key_map), None)
        if key is None:
            raise ValueError("Confusion dict must contain a 'confusion' key.")
        conf_obj = conf_obj[key]
    confusion = _to_numpy(conf_obj)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError(f"Confusion matrix must be square. Got shape {confusion.shape}.")
    return confusion


def compute_confusion_from_arrays(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, ignore_indices: Iterable[int]
) -> np.ndarray:
    y_true_flat = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred_flat = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    ignore_set = set(ignore_indices)
    mask = (~np.isin(y_true_flat, list(ignore_set))) & (~np.isin(y_pred_flat, list(ignore_set)))
    y_true_flat = y_true_flat[mask]
    y_pred_flat = y_pred_flat[mask]
    valid_mask = (
        (y_true_flat >= 0)
        & (y_true_flat < num_classes)
        & (y_pred_flat >= 0)
        & (y_pred_flat < num_classes)
    )
    y_true_flat = y_true_flat[valid_mask]
    y_pred_flat = y_pred_flat[valid_mask]
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (y_true_flat, y_pred_flat), 1)
    dropped = int(mask.size - valid_mask.sum())
    if dropped:
        print(f"Skipped {dropped} predictions due to ignore set or invalid indices.")
    return confusion


def species_confusion_from_matrix(
    base_confusion: np.ndarray,
    entries: List[Dict[str, str]],
    ignore_indices: Iterable[int],
) -> Tuple[np.ndarray, List[str]]:
    sorted_indices, sorted_labels = sort_species_by_genus(entries, ignore_indices)
    species_confusion = base_confusion[np.ix_(sorted_indices, sorted_indices)]
    return species_confusion, sorted_labels


def aggregate_genus_confusion_from_matrix(
    base_confusion: np.ndarray,
    entries: List[Dict[str, str]],
    ignore_indices: Iterable[int],
) -> Tuple[np.ndarray, List[str]]:
    ignore_set = set(ignore_indices)
    label_to_genus: Dict[int, str] = {idx: entry["genus"] for idx, entry in enumerate(entries) if entry["genus"]}
    genus_names = sorted(set(label_to_genus.values()))
    genus_to_idx = {name: i for i, name in enumerate(genus_names)}
    genus_confusion = np.zeros((len(genus_names), len(genus_names)), dtype=np.int64)
    for true_idx, row in enumerate(base_confusion):
        if true_idx in ignore_set:
            continue
        genus_true = label_to_genus.get(true_idx)
        if not genus_true:
            continue
        true_genus_idx = genus_to_idx[genus_true]
        for pred_idx, value in enumerate(row):
            if value == 0 or pred_idx in ignore_set:
                continue
            genus_pred = label_to_genus.get(pred_idx)
            if not genus_pred:
                continue
            pred_genus_idx = genus_to_idx[genus_pred]
            genus_confusion[true_genus_idx, pred_genus_idx] += int(value)
    return genus_confusion, genus_names


def save_confusion_csv(matrix: np.ndarray, labels: Sequence[str], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + list(labels))
        for label, row in zip(labels, matrix):
            writer.writerow([label] + [int(v) for v in row])


def plot_confusion(matrix: np.ndarray, labels: Sequence[str], path: Path, title: str) -> None:
    fig, ax = plt.subplots(
        figsize=(max(8.0, 0.25 * len(labels)), max(6.0, 0.25 * len(labels))),
        constrained_layout=True,
    )
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ticks = np.arange(len(labels))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_yticklabels(labels, fontsize=5)
    plt.savefig(path, dpi=300)
    plt.close(fig)


def save_confusion_variants(
    matrix: np.ndarray,
    labels: Sequence[str],
    output_dir: Path,
    prefix: str,
    title: str,
    skip_plots: bool,
) -> None:
    np.save(output_dir / f"{prefix}.npy", matrix)
    save_confusion_csv(matrix, labels, output_dir / f"{prefix}.csv")
    if not skip_plots:
        plot_confusion(matrix, labels, output_dir / f"{prefix}.png", title)


def build_confusion_matrices(
    preds_path: Optional[Path],
    targets_path: Optional[Path],
    confusion_path: Optional[Path],
    output_dir: Path,
    plots_prefix: str,
    skip_plots: bool,
    ignore_indices: Iterable[int],
    entries: List[Dict[str, str]],
) -> None:
    num_classes = len(entries)
    ignore_set = set(ignore_indices)

    if confusion_path:
        base_confusion = load_confusion_matrix(confusion_path)
        if base_confusion.shape != (num_classes, num_classes):
            raise ValueError(
                f"Confusion matrix shape {base_confusion.shape} does not match number of classes {num_classes}."
            )
        base_confusion = np.asarray(base_confusion, dtype=np.int64)
        source_path: Path = confusion_path
    else:
        if preds_path is None or targets_path is None:
            raise ValueError("Provide --preds and --targets or --confusion.")
        preds, targets = load_preds_and_targets(preds_path, targets_path)
        base_confusion = compute_confusion_from_arrays(
            y_true=targets, y_pred=preds, num_classes=num_classes, ignore_indices=ignore_set
        )
        source_path = preds_path

    species_confusion, species_labels = species_confusion_from_matrix(
        base_confusion, entries, ignore_indices=ignore_set
    )
    genus_confusion, genus_labels = aggregate_genus_confusion_from_matrix(
        base_confusion, entries, ignore_indices=ignore_set
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_confusion_variants(
        species_confusion,
        species_labels,
        output_dir,
        f"{plots_prefix}_species",
        "Species confusion (sorted by genus)",
        skip_plots,
    )
    save_confusion_variants(
        genus_confusion,
        genus_labels,
        output_dir,
        f"{plots_prefix}_genus",
        "Genus confusion (alphabetical)",
        skip_plots,
    )

    plot_suffix = "" if skip_plots else "/.png"
    print(f"Species-level confusion -> {output_dir}/{plots_prefix}_species[.npy/.csv{plot_suffix}]")
    print(f"Genus-level confusion   -> {output_dir}/{plots_prefix}_genus[.npy/.csv{plot_suffix}]")
    print(f"Source for counts: {source_path}")
    if ignore_set:
        print(f"Ignored class indices: {sorted(ignore_set)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build taxonomy distance matrix and/or taxonomy-ordered confusion matrices."
    )
    parser.add_argument("--preds", type=Path, help="Path to predictions array.")
    parser.add_argument("--targets", type=Path, help="Path to ground-truth labels.")
    parser.add_argument("--confusion", type=Path, help="Existing species-level confusion matrix in class index order.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(BASE_DIR),
        help="Directory to save confusion matrices and plots.",
    )
    parser.add_argument(
        "--plots-prefix",
        type=str,
        default="taxonomy",
        help="Prefix for saved confusion matrices (e.g., taxonomy_species.png).",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Save .npy/.csv but skip rendering PNG plots (useful for headless runs).",
    )
    parser.add_argument(
        "--rebuild-distance",
        action="store_true",
        help="Also rebuild the taxonomy distance matrix PT/CSV alongside confusion outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = prepare_entries()

    built_confusion = False
    if args.confusion or args.preds or args.targets:
        build_confusion_matrices(
            preds_path=args.preds,
            targets_path=args.targets,
            confusion_path=args.confusion,
            output_dir=args.output_dir,
            plots_prefix=args.plots_prefix,
            skip_plots=args.skip_plots,
            ignore_indices=IGNORE_INDICES,
            entries=entries,
        )
        built_confusion = True

    if args.rebuild_distance or not built_confusion:
        save_taxonomy_distance_matrix(entries)


if __name__ == "__main__":
    main()
