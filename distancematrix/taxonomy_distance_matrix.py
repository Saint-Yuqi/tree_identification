import csv
import os
from typing import Dict, List, Sequence

import torch

ROOT_NODE = "root"
LEVELS = ("class", "order", "family", "genus")
BASE_DIR = os.path.dirname(__file__)

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


def main() -> None:
    entries = prepare_entries()
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


if __name__ == "__main__":
    main()
