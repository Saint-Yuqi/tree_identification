"""
Distance-bucket analysis for taxonomy-aware errors.

Given predictions and ground-truth labels (or a confusion matrix), this script
computes how misclassifications are distributed across taxonomy distance
intervals for two models (Baseline vs. Hierarchical).

Example (pred/target arrays):
    python analyse_data/distance_bucket_analysis.py \
        --baseline-preds /path/baseline_preds.npy \
        --baseline-targets /path/baseline_targets.npy \
        --ours-preds /path/ours_preds.npy \
        --ours-targets /path/ours_targets.npy

Example (confusion matrices):
    python analyse_data/distance_bucket_analysis.py \
        --baseline-confusion /path/baseline_confusion.npy \
        --ours-confusion /path/ours_confusion.npy
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Bucket = Tuple[str, float, float]

# Default buckets: inclusive of bounds
DEFAULT_BUCKETS: Sequence[Bucket] = (
    ("near_0_2", 0.0, 2.0),
    ("mid_3_6", 3.0, 6.0),
    ("far_7_10", 7.0, 10.0),
)


def load_distance_matrix(csv_path: Path) -> Tuple[np.ndarray, List[str]]:
    """Load taxonomy distance matrix from CSV generated in distancematrix/."""
    labels: List[str] = []
    rows: List[List[float]] = []
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        col_labels = header[1:] if header else []
        for row in reader:
            if not row:
                continue
            labels.append(row[0])
            rows.append([float(v) for v in row[1:]])
    matrix = np.asarray(rows, dtype=np.float32)
    # Column labels are identical to row labels in this repo
    return matrix, (col_labels if col_labels else labels)


def _to_numpy(array_like) -> np.ndarray:
    """Convert arrays or tensors to numpy."""
    if isinstance(array_like, np.ndarray):
        return array_like
    try:
        import torch

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
        import torch

        return torch.load(path, map_location="cpu")
    if ext in (".csv", ".txt"):
        return np.loadtxt(path, delimiter=",")
    raise ValueError(f"Unsupported file extension for {path}")


def _extract_preds_targets(container: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    key_map = {k.lower(): k for k in container.keys()}
    pred_key = next((key_map[k] for k in ("preds", "predictions", "outputs", "logits") if k in key_map), None)
    target_key = next((key_map[k] for k in ("targets", "labels", "gts", "gt", "masks") if k in key_map), None)
    if pred_key is None or target_key is None:
        raise ValueError("Could not find prediction/target keys in container.")
    return _to_numpy(container[pred_key]), _to_numpy(container[target_key])


def load_preds_and_targets(pred_path: Path, target_path: Optional[Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Load prediction and target arrays from either a pair of files or one container file."""
    pred_obj = load_array(pred_path)
    if isinstance(pred_obj, dict) and target_path is None:
        preds, targets = _extract_preds_targets(pred_obj)
    elif target_path is not None:
        preds = _to_numpy(pred_obj)
        targets = _to_numpy(load_array(target_path))
    else:
        raise ValueError("Provide both --*_preds and --*_targets or a single container file with both arrays.")
    if preds.shape != targets.shape:
        raise ValueError(f"Shape mismatch between preds {preds.shape} and targets {targets.shape}.")
    return preds, targets


def load_confusion(path: Path) -> np.ndarray:
    conf_obj = load_array(path)
    if isinstance(conf_obj, dict):
        key_map = {k.lower(): k for k in conf_obj.keys()}
        key = next((key_map[k] for k in ("confusion", "conf", "cm") if k in key_map), None)
        if key is None:
            raise ValueError("Confusion dict must contain a 'confusion' key.")
        return _to_numpy(conf_obj[key])
    conf_arr = _to_numpy(conf_obj)
    if conf_arr.ndim != 2 or conf_arr.shape[0] != conf_arr.shape[1]:
        raise ValueError(f"Confusion matrix must be square. Got shape {conf_arr.shape}.")
    return conf_arr


def bucket_name(distance: float, buckets: Sequence[Bucket]) -> str:
    for name, low, high in buckets:
        if low <= distance <= high:
            return name
    return buckets[-1][0]


def summarize_from_distances(distances: np.ndarray, buckets: Sequence[Bucket]) -> Dict[str, float]:
    counts = {name: 0 for name, _, _ in buckets}
    for d in distances:
        counts[bucket_name(float(d), buckets)] += 1
    total_errors = int(distances.size)
    weighted = float(distances.sum())
    return summarize_from_counts(counts, total_errors, weighted, buckets)


def summarize_from_counts(
    bucket_counts: Dict[str, int], total_errors: int, weighted_sum: float, buckets: Sequence[Bucket]
) -> Dict[str, float]:
    percentages = {
        name: (bucket_counts[name] / total_errors * 100.0 if total_errors else 0.0) for name, _, _ in buckets
    }
    return {
        "total_errors": int(total_errors),
        "mean_distance": weighted_sum / total_errors if total_errors else 0.0,
        "counts": bucket_counts,
        "percentages": percentages,
    }


def analyze_from_arrays(
    preds: np.ndarray,
    targets: np.ndarray,
    distance_matrix: np.ndarray,
    buckets: Sequence[Bucket],
    ignore_indices: Optional[Iterable[int]] = None,
) -> Dict[str, float]:
    preds_flat = preds.reshape(-1).astype(np.int64)
    targets_flat = targets.reshape(-1).astype(np.int64)
    mask = np.ones_like(preds_flat, dtype=bool)
    if ignore_indices is not None:
        ignore_set = set(ignore_indices)
        for idx in ignore_set:
            mask &= targets_flat != idx
    num_classes = distance_matrix.shape[0]
    mask &= (preds_flat >= 0) & (preds_flat < num_classes) & (targets_flat >= 0) & (targets_flat < num_classes)
    preds_flat = preds_flat[mask]
    targets_flat = targets_flat[mask]
    error_mask = preds_flat != targets_flat
    if not np.any(error_mask):
        return summarize_from_distances(np.array([]), buckets)
    distances = distance_matrix[targets_flat[error_mask], preds_flat[error_mask]]
    return summarize_from_distances(distances, buckets)


def analyze_from_confusion(
    confusion: np.ndarray,
    distance_matrix: np.ndarray,
    buckets: Sequence[Bucket],
    ignore_indices: Optional[Iterable[int]] = None,
) -> Dict[str, float]:
    num_classes = distance_matrix.shape[0]
    if confusion.shape != (num_classes, num_classes):
        raise ValueError(f"Confusion matrix shape {confusion.shape} does not match distance matrix {distance_matrix.shape}.")
    ignore_set = set(ignore_indices) if ignore_indices is not None else set()
    bucket_counts = {name: 0 for name, _, _ in buckets}
    total_errors = 0
    weighted_sum = 0.0
    for true_idx in range(num_classes):
        if true_idx in ignore_set:
            continue
        for pred_idx in range(num_classes):
            if pred_idx in ignore_set or pred_idx == true_idx:
                continue
            count = int(confusion[true_idx, pred_idx])
            if count <= 0:
                continue
            distance = float(distance_matrix[true_idx, pred_idx])
            bucket_counts[bucket_name(distance, buckets)] += count
            total_errors += count
            weighted_sum += distance * count
    return summarize_from_counts(bucket_counts, total_errors, weighted_sum, buckets)


def analyze_model(
    name: str,
    distance_matrix: np.ndarray,
    buckets: Sequence[Bucket],
    preds_path: Optional[Path],
    targets_path: Optional[Path],
    confusion_path: Optional[Path],
    ignore_indices: Optional[List[int]],
) -> Dict[str, Dict[str, float]]:
    if confusion_path:
        confusion = load_confusion(confusion_path)
        summary = analyze_from_confusion(confusion, distance_matrix, buckets, ignore_indices)
        source = confusion_path
    elif preds_path:
        preds, targets = load_preds_and_targets(preds_path, targets_path)
        summary = analyze_from_arrays(preds, targets, distance_matrix, buckets, ignore_indices)
        source = preds_path
    else:
        raise ValueError(f"{name}: provide either predictions/targets or a confusion matrix.")
    summary["source"] = str(source)
    return {name: summary}


def format_row(model: str, summary: Dict[str, float], buckets: Sequence[Bucket]) -> str:
    counts = summary["counts"]
    percentages = summary["percentages"]
    pieces = [
        f"{model:<14}",
        f"err={summary['total_errors']:>8}",
        f"meanD={summary['mean_distance']:.2f}",
    ]
    for name, low, high in buckets:
        pieces.append(f"{name}:{percentages[name]:5.1f}% ({counts[name]})")
    return " | ".join(pieces)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distance-bucket analysis for taxonomy-aware errors.")
    parser.add_argument("--distance-matrix", type=Path, default=Path("distancematrix/taxonomy_distance_matrix.csv"))
    parser.add_argument("--baseline-preds", type=Path, help="Predictions file for baseline model.")
    parser.add_argument("--baseline-targets", type=Path, help="Targets file for baseline model.")
    parser.add_argument("--baseline-confusion", type=Path, help="Confusion matrix file for baseline model.")
    parser.add_argument("--ours-preds", type=Path, help="Predictions file for hierarchical model.")
    parser.add_argument("--ours-targets", type=Path, help="Targets file for hierarchical model.")
    parser.add_argument("--ours-confusion", type=Path, help="Confusion matrix file for hierarchical model.")
    parser.add_argument("--ignore-index", type=int, action="append", help="Label indices to ignore (can be set multiple times).")
    parser.add_argument("--save-json", type=Path, help="Optional path to dump the numeric summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    distance_matrix, labels = load_distance_matrix(args.distance_matrix)
    ignore_indices = args.ignore_index if args.ignore_index else None

    baseline = analyze_model(
        "Baseline",
        distance_matrix,
        DEFAULT_BUCKETS,
        preds_path=args.baseline_preds,
        targets_path=args.baseline_targets,
        confusion_path=args.baseline_confusion,
        ignore_indices=ignore_indices,
    )
    ours = analyze_model(
        "Ours",
        distance_matrix,
        DEFAULT_BUCKETS,
        preds_path=args.ours_preds,
        targets_path=args.ours_targets,
        confusion_path=args.ours_confusion,
        ignore_indices=ignore_indices,
    )

    summaries = {**baseline, **ours, "distance_matrix": str(args.distance_matrix), "class_labels": labels}

    print("\nDistance-bucket share of errors (0–10 taxonomy distance)\n")
    for model_name in ("Baseline", "Ours"):
        print(format_row(model_name, summaries[model_name], DEFAULT_BUCKETS))
    print(f"\nDistance matrix: {args.distance_matrix}")
    if ignore_indices:
        print(f"Ignored indices: {sorted(set(ignore_indices))}")
    print(f"Saved source paths -> Baseline: {summaries['Baseline']['source']}, Ours: {summaries['Ours']['source']}")

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with args.save_json.open("w") as f:
            json.dump(summaries, f, indent=2)
        print(f"\nWrote summary to {args.save_json}")


if __name__ == "__main__":
    main()
