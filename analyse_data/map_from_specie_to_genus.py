import yaml
import os
from typing import Sequence


""" this file creates a map from specie to genus 
This is used to make the confusion matrix on genus level"""



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



genus_set = []
specie_to_genus_index = {}
genus_to_index = {}

specieindex = 0
for species, _, _, _, genus in TAXONOMY_ROWS:
    
    # If genus not given, use species name
    if genus == "":
        genus = species

    if genus not in genus_to_index:
        genus_to_index[genus] = len(genus_to_index)   # start at 0

    specie_to_genus_index[specieindex] = genus_to_index[genus]
    specieindex =specieindex +1


#save file
output = {
    "genus_to_index": genus_to_index,
    "specie_to_genus_index": specie_to_genus_index,
}

save_path = os.path.join("..","configs", "genus", "treeAI_genus.yaml")

with open(save_path, "w") as f:
    yaml.safe_dump(output, f, sort_keys=False)

print(f"Saved {save_path}")