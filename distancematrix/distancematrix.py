import dendropy
import os
import pandas as pd
import torch
import numpy as np

newick_str = """
((((betula_papyrifera:0.05,betula_alleghaniensis:0.05,betula_pendula:0.05,betula_lenta:0.05)Betula:0.1,
    (alnus_glutinosa:0.08)Alnus:0.12,
    (carpinus_betulus:0.08)Carpinus:0.12)Betulaceae:0.15,
   ((acer_saccharum:0.03,acer_pseudoplatanus:0.03,acer_platanoides:0.03,acer_rubrum:0.03,acer_pensylvanicum:0.03,acer_negundo:0.03,acer_campestre:0.03)Acer:0.1)Sapindaceae:0.15,
   ((quercus_robur:0.03,quercus_ilex:0.03,quercus_rubra:0.03,quercus_petraea:0.03,quercus_alba:0.03)Quercus:0.1,
    (fagus_grandifolia:0.05,fagus_sylvatica:0.05,fagus_crenata:0.05)Fagus:0.1,
    (castanea_sativa:0.08)Castanea:0.12)Fagaceae:0.15,
   ((populus_balsamifera:0.05,populus_grandidentata:0.05,populus_tremula:0.05)Populus:0.1,
    (salix_alba:0.08,salix_babylonica:0.08)Salix:0.12)Salicaceae:0.15,
   ((fraxinus_nigra:0.05,fraxinus_excelsior:0.05)Fraxinus:0.1)Oleaceae:0.15,
   ((tilia_europaea:0.05,tilia_cordata:0.05)Tilia:0.1)Malvaceae:0.15,
   (robinia_pseudoacacia:0.1)Fabaceae:0.2,
   (crataegus_monogyna:0.1)Rosaceae:0.2)Angiosperms:0.3,
  (((pinus_sylvestris:0.03,pinus_strobus:0.03,pinus_pinea:0.03,pinus_montezumae:0.03,pinus_elliottii:0.03,pinus_nigra:0.03,pinus_koraiensis:0.03,pinus_ponderosa:0.03)Pinus:0.1,
    (picea_abies:0.03,picea_rubens:0.03,picea_mariana:0.03,picea_glauca:0.03)Picea:0.1,
    (abies_alba:0.03,abies_holophylla:0.03,abies_firma:0.03,abies_balsamea:0.03)Abies:0.1,
    (larix_decidua:0.03,larix_laricina:0.03,larix_gmelinii:0.03)Larix:0.1,
    (pseudotsuga_menziesii:0.08)Pseudotsuga:0.12,
    (tsuga_canadensis:0.08)Tsuga:0.12)Pinaceae:0.2,
   ((cryptomeria_japonica:0.05,thuja_occidentalis:0.05)Cupressaceae_main:0.1,
    (juniperus_communis:0.08)Cupressaceae_secondary:0.12)Cupressaceae:0.25,
   (cedrus_libani:0.15)Pinaceae_Cedar:0.25)Conifers:0.4,
  (dacrydium_cupressinum:0.5)Podocarpaceae:0.6,
  (eucalyptus_globulus:0.4,metrosideros_umbellata:0.4)Myrtaceae:0.5);
"""

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
    "betula_sp.": "betula_lenta",           
    "picea_sp.": "picea_glauca",           
    "acer_sp.": "acer_negundo",            
    "pinus_sp.": "pinus_ponderosa",        
    "carpinus_sp.": "carpinus_betulus",    
    "populus_sp.": "populus_tremula",      
    "quercus_sp.": "quercus_alba",         
    "coniferous": "juniperus_communis",    
    "deciduous": "acer_campestre",         
    "salix_sp.": "salix_babylonica"        
}

labels_species = [representatives.get(l, l) for l in labels]

phylo_tree = dendropy.Tree.get(data=newick_str, schema="newick", preserve_underscores=True)
pdm = phylo_tree.phylogenetic_distance_matrix()

num_classes = len(labels)
dist_matrix = pd.DataFrame(index=labels, columns=labels, dtype=float)

tree_species = [taxon.label for taxon in phylo_tree.taxon_namespace]
print("Species in tree:", len(tree_species))
print("Species in labels:", len(set(labels_species)))

missing_species = set(labels_species) - set(tree_species) - {"Background", "dead_tree"}
if missing_species:
    print(f"Missing from tree: {missing_species}")
else:
    print("All species found in tree!")

for i, s1 in enumerate(labels_species):
    for j, s2 in enumerate(labels_species):
        # Handle Background class
        if labels[i] == "Background" and labels[j] == "Background":
            dist_matrix.iloc[i, j] = 0.0
        elif labels[i] == "Background" and labels[j] == "dead_tree":
            dist_matrix.iloc[i, j] = 1.0
        elif labels[j] == "Background" and labels[i] == "dead_tree":
            dist_matrix.iloc[i, j] = 1.0
        elif labels[i] == "Background" or labels[j] == "Background":
            dist_matrix.iloc[i, j] = 2.0
        # Handle dead_tree class
        elif labels[i] == "dead_tree" and labels[j] == "dead_tree":
            dist_matrix.iloc[i, j] = 0.0
        elif labels[i] == "dead_tree" or labels[j] == "dead_tree":
            dist_matrix.iloc[i, j] = 1.0
        else:
            try:
                node1 = phylo_tree.find_node_with_taxon_label(s1)
                node2 = phylo_tree.find_node_with_taxon_label(s2)
                
                if node1 is None or node2 is None:
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

dist_matrix = dist_matrix.round(2)
D = torch.tensor(dist_matrix.values, dtype=torch.float)

# Ensure files are saved in the same directory as this script
base_dir = os.path.dirname(__file__)

torch.save(D, os.path.join(base_dir, 'phylogenetic_distance_matrix.pt'))
print(f"Distance matrix saved as tensor with shape {D.shape}")

dist_matrix.to_csv(os.path.join(base_dir, 'phylogenetic_distance_matrix.csv'), float_format='%.2f')
print(f"Distance matrix saved as CSV with shape {dist_matrix.shape}")

torch.save(labels, os.path.join(base_dir, 'class_labels.pt'))
with open(os.path.join(base_dir, 'class_labels.txt'), 'w') as f:
    for i, label in enumerate(labels):
        f.write(f"{i}: {label}\n")
print("Labels saved for reference")

print(f"\nDistance Matrix Statistics:")
dead_tree_idx = labels.index('dead_tree')
background_idx = labels.index('Background')
betula_pap_idx = labels.index('betula_papyrifera')

print(f"Background <-> betula_papyrifera: {D[background_idx, betula_pap_idx]:.2f}")
print(f"Background <-> Dead Tree: {D[background_idx, dead_tree_idx]:.2f}")
print(f"Dead Tree <-> betula_papyrifera: {D[dead_tree_idx, betula_pap_idx]:.2f}")

print(f"\nFiles saved:")
print("- phylogenetic_distance_matrix.pt")
print("- phylogenetic_distance_matrix.csv") 
print("- class_labels.pt")
print("- class_labels.txt")