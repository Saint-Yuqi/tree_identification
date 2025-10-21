import dendropy
import pandas as pd
import torch

import dendropy
import pandas as pd
import torch

# Add this comprehensive Newick string before the phylo_tree line:
newick_str = """
((((betula_papyrifera:0.05,betula_alleghaniensis:0.05,betula_pendula:0.05)Betula:0.1,
    (alnus_glutinosa:0.08)Alnus:0.12,
    (carpinus_betulus:0.08)Carpinus:0.12)Betulaceae:0.15,
   ((acer_saccharum:0.03,acer_pseudoplatanus:0.03,acer_platanoides:0.03,acer_rubrum:0.03,acer_pensylvanicum:0.03)Acer:0.1)Sapindaceae:0.15,
   ((quercus_robur:0.03,quercus_ilex:0.03,quercus_rubra:0.03,quercus_petraea:0.03)Quercus:0.1,
    (fagus_grandifolia:0.05,fagus_sylvatica:0.05,fagus_crenata:0.05)Fagus:0.1,
    (castanea_sativa:0.08)Castanea:0.12)Fagaceae:0.15,
   ((populus_balsamifera:0.05,populus_grandidentata:0.05)Populus:0.1,
    (salix_alba:0.08)Salix:0.12)Salicaceae:0.15,
   ((fraxinus_nigra:0.05,fraxinus_excelsior:0.05)Fraxinus:0.1)Oleaceae:0.15,
   ((tilia_europaea:0.05,tilia_cordata:0.05)Tilia:0.1)Malvaceae:0.15,
   (robinia_pseudoacacia:0.1)Fabaceae:0.2,
   (crataegus_monogyna:0.1)Rosaceae:0.2)Angiosperms:0.3,
  (((pinus_sylvestris:0.03,pinus_strobus:0.03,pinus_pinea:0.03,pinus_montezumae:0.03,pinus_elliottii:0.03,pinus_nigra:0.03,pinus_koraiensis:0.03)Pinus:0.1,
    (picea_abies:0.03,picea_rubens:0.03,picea_mariana:0.03)Picea:0.1,
    (abies_alba:0.03,abies_holophylla:0.03,abies_firma:0.03,abies_balsamea:0.03)Abies:0.1,
    (larix_decidua:0.03,larix_laricina:0.03,larix_gmelinii:0.03)Larix:0.1,
    (pseudotsuga_menziesii:0.08)Pseudotsuga:0.12,
    (tsuga_canadensis:0.08)Tsuga:0.12)Pinaceae:0.2,
   (cryptomeria_japonica:0.15)Cupressaceae:0.25,
   (cedrus_libani:0.15)Pinaceae_Cedar:0.25)Conifers:0.4,
  (dacrydium_cupressinum:0.5)Podocarpaceae:0.6,
  (eucalyptus_globulus:0.4,metrosideros_umbellata:0.4)Myrtaceae:0.5);
"""

# ...existing code...
# -----------------------------
# Your labels and representative mapping
# -----------------------------
labels = [
    "Background", "betula_papyrifera", "tsuga_canadensis", "picea_abies",
    "acer_saccharum", "betula_sp.", "pinus_sylvestris", "picea_rubens",
    "betula_alleghaniensis", "larix_decidua", "fagus_grandifolia", "picea_sp.",
    "fagus_sylvatica", "dead_tree", "acer_pensylvanicum", "populus_balsamifera",
    "quercus_ilex", "quercus_robur", "pinus_strobus", "larix_laricina",
    "larix_gmelinii", "pinus_pinea", "populus_grandidentata", "pinus_montezumae",
    "abies_alba", "betula_pendula", "pseudotsuga_menziesii", "fraxinus_nigra",
    "dacrydium_cupressinum", "cedrus_libani", "acer_pseudoplatanus", "pinus_elliottii",
    "cryptomeria_japonica", "pinus_koraiensis", "abies_holophylla", "alnus_glutinosa",
    "fraxinus_excelsior", "coniferous", "eucalyptus_globulus", "pinus_nigra",
    "quercus_rubra", "tilia_europaea", "abies_firma", "acer_sp.", "metrosideros_umbellata",
    "acer_rubrum", "picea_mariana", "abies_balsamea", "castanea_sativa", "tilia_cordata",
    "populus_sp.", "crataegus_monogyna", "quercus_petraea", "acer_platanoides",
    "robinia_pseudoacacia", "fagus_crenata", "quercus_sp.", "salix_alba", "pinus_sp.",
    "carpinus_sp.", "deciduous", "salix_sp."
]

representatives = {
    "betula_sp.": "betula_papyrifera",
    "picea_sp.": "picea_abies",
    "acer_sp.": "acer_saccharum",
    "pinus_sp.": "pinus_sylvestris",
    "carpinus_sp.": "carpinus_betulus",
    "populus_sp.": "populus_balsamifera",
    "quercus_sp.": "quercus_robur",
    "coniferous": "picea_abies",
    "deciduous": "acer_saccharum",
    "dead_tree": "betula_papyrifera",
    "salix_sp.": "salix_alba"
}

# Map all labels to species
labels_species = [representatives.get(l, l) for l in labels]

# -----------------------------
# Load tree and compute distance
# -----------------------------
phylo_tree = dendropy.Tree.get(data=newick_str, schema="newick", preserve_underscores=True)
pdm = phylo_tree.phylogenetic_distance_matrix()

# Initialize distance matrix
num_classes = len(labels)
dist_matrix = pd.DataFrame(index=labels, columns=labels, dtype=float)

# Check which species are missing from the tree
tree_species = [taxon.label for taxon in phylo_tree.taxon_namespace]
print("Species in tree:", len(tree_species))
print("Species in labels:", len(set(labels_species)))

missing_species = set(labels_species) - set(tree_species) - {"Background"}
if missing_species:
    print(f"Missing from tree: {missing_species}")

for i, s1 in enumerate(labels_species):
    for j, s2 in enumerate(labels_species):
        if labels[i] == "Background" or labels[j] == "Background":
            dist_matrix.iloc[i, j] = 2.0
        else:
            try:
                node1 = phylo_tree.find_node_with_taxon_label(s1)
                node2 = phylo_tree.find_node_with_taxon_label(s2)
                
                if node1 is None or node2 is None:
                    # Species not found in tree, assign maximum distance
                    dist_matrix.iloc[i, j] = 1.0
                    if node1 is None:
                        print(f"Species not found in tree: {s1}")
                    if node2 is None:
                        print(f"Species not found in tree: {s2}")
                else:
                    t1 = node1.taxon
                    t2 = node2.taxon
                    dist_matrix.iloc[i, j] = pdm.distance(t1, t2)
            except Exception as e:
                print(f"Error processing {s1} vs {s2}: {e}")
                dist_matrix.iloc[i, j] = 1.0

# Convert to PyTorch tensor
D = torch.tensor(dist_matrix.values, dtype=torch.float)

# Save the tensor for use in other scripts
torch.save(D, 'phylogenetic_distance_matrix.pt')
print(f"Distance matrix saved as tensor with shape {D.shape}")

# Also save the labels for reference
torch.save(labels, 'class_labels.pt')
print("Labels saved for reference")