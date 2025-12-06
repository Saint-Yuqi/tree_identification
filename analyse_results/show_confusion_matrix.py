import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


'''this file generates the confusion matrixes for species and genus
these are used for the paper
inputs: confusion matrix from test file (change the path!)'''

#load data:
conf_genus_path = "/home/c/shursc/code/tree_identification/lightning_logs/20251119_093300.602560_Unet_resnet50_CE_320_62_wrs_True/test_20251206_100416.557309_best_epoch=315_640/quantitative/top_1/confusion_genus.npy"
conf_genus_hierarchical_path = "/home/c/shursc/code/tree_identification/lightning_logs/20251120_093110.096873_Unet_resnet50_HierarchicalCE_320_62_wrs_True/test_20251206_113046.771390_best_epoch=469_640/quantitative/top_1/confusion_genus.npy"
conf_genus= np.load(conf_genus_path)
conf_genus_hier =np.load(conf_genus_hierarchical_path)


# Original genus labels
genus_to_index = {
    "Background": 0, "Betula": 1, "Tsuga": 2, "Picea": 3, "Acer": 4, "Pinus": 5, "Larix": 6,
    "Fagus": 7, "dead tree": 8, "Populus": 9, "Quercus": 10, "Abies": 11, "Pseudotsuga": 12,
    "Fraxinus": 13, "Dacrydium": 14, "Cedrus": 15, "Cryptomeria": 16, "Alnus": 17, 
    "coniferous": 18, "Eucalyptus": 19, "Tilia": 20, "Metrosideros": 21, "Castanea": 22, 
    "Crataegus": 23, "Robinia": 24, "Salix": 25, "Carpinus": 26, "deciduous": 27
}

#remove indexes that we dont need to show
ignore_index_genus=[18,26,27] #corniferous, carpinus, deciduous 
#(these lines are already 0, otherwise we wouold get wrong results!!!)
labels = [name for name, idx in sorted(genus_to_index.items(), key=lambda x: x[1]) 
          if idx not in ignore_index_genus]

#remove labels from original confusion matrix (here it is important that the corresponding rows/columns are 0!)
conf_genus = np.delete(conf_genus, ignore_index_genus, axis=0)
conf_genus = np.delete(conf_genus, ignore_index_genus, axis=1)

conf_genus_hier= np.delete(conf_genus_hier, ignore_index_genus, axis=0)
conf_genus_hier = np.delete(conf_genus_hier, ignore_index_genus, axis=1)

conf_diff = conf_genus_hier - conf_genus

# Plot confusion matrix for genus level
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(conf_genus, display_labels=labels)
disp.plot(ax=ax, cmap='Blues', values_format='.0f',colorbar=False)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=11)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=11)
for text in ax.texts:
    text.set_fontsize(8)

# Rotate x-axis labels
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
ax.set_xlabel("Predicted label", fontsize=13)
ax.set_ylabel("True label", fontsize=13)
plt.title("Genus Confusion Matrix", fontsize=15)
plt.tight_layout()
#plt.savefig("genus_confusion.png")
plt.close()



#DIFFERENCE
fig, ax = plt.subplots(figsize=(7, 6))

# Display the difference matrix
im = ax.imshow(conf_diff, cmap='bwr_r', interpolation='nearest', vmin=-50, vmax=50)

# Set ticks and labels
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
ax.set_yticklabels(labels, fontsize=11)

# Add black text annotations
for i in range(conf_diff.shape[0]):
    for j in range(conf_diff.shape[1]):
        ax.text(j, i, f'{conf_diff[i, j]:.0f}', ha='center', va='center', color='black', fontsize=9)

# Axis labels and title
ax.set_xlabel("Predicted label", fontsize=13)
ax.set_ylabel("True label", fontsize=13)
ax.set_title("Difference in Confusion Matrices", fontsize=16)

plt.tight_layout()
plt.savefig("genus_confusion_difference.pdf", dpi=300)
plt.close()



#############################################
#  plot with three matrices: BL, Hierarchical and Difference 
##############################################

#plot with two subplots:

# Plot 3 subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Model 1
disp1 = ConfusionMatrixDisplay(conf_genus, display_labels=labels)
disp1.plot(ax=axes[0], cmap='Blues', values_format='.0f', colorbar=False)
axes[0].set_title("BL Model", fontsize=15)
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right', fontsize=12)
axes[0].set_yticklabels(axes[0].get_yticklabels(), fontsize=12)

# Model 2 (Hierarchical)
disp2 = ConfusionMatrixDisplay(conf_genus_hier, display_labels=labels)
disp2.plot(ax=axes[1], cmap='Blues', values_format='.0f', colorbar=False)
axes[1].set_title("Hierarchical Model", fontsize=15)
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right', fontsize=12)
axes[1].set_yticklabels(axes[1].get_yticklabels(), fontsize=12)


# Common labels
for ax in axes:
    ax.set_xlabel("Predicted label", fontsize=13)
    ax.set_ylabel("True label", fontsize=13)

plt.tight_layout()
plt.savefig("genus_confusion_BL_hierarchical.pdf", dpi=300, bbox_inches='tight')



#############################################
#  plot with three matrices: BL, Hierarchical and Difference 
##############################################

fig, axes = plt.subplots(3, 1, figsize=(12, 20))

# Model 1
disp1 = ConfusionMatrixDisplay(conf_genus, display_labels=labels)
disp1.plot(ax=axes[0], cmap='Blues', values_format='.0f', colorbar=False)
axes[0].set_title("BL Model", fontsize=16)
axes[0].set_xlabel("Predicted label", fontsize=14)
axes[0].set_ylabel("True label", fontsize=14)
axes[0].set_xticks(np.arange(len(labels)))
axes[0].set_yticks(np.arange(len(labels)))
axes[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
axes[0].set_yticklabels(labels, fontsize=12)

# Model 2
disp2 = ConfusionMatrixDisplay(conf_genus_hier, display_labels=labels)
disp2.plot(ax=axes[1], cmap='Blues', values_format='.0f', colorbar=False)
axes[1].set_title("Hierarchical Model", fontsize=16)
axes[1].set_xlabel("Predicted label", fontsize=14)
axes[1].set_ylabel("True label", fontsize=14)
axes[1].set_xticks(np.arange(len(labels)))
axes[1].set_yticks(np.arange(len(labels)))
axes[1].set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
axes[1].set_yticklabels(labels, fontsize=12)

# Difference
im = axes[2].imshow(conf_diff, cmap='bwr_r', interpolation='nearest', vmin=-50, vmax=50)
axes[2].set_title("Difference", fontsize=16)
axes[2].set_xlabel("Predicted label", fontsize=14)
axes[2].set_ylabel("True label", fontsize=14)
axes[2].set_xticks(np.arange(len(labels)))
axes[2].set_yticks(np.arange(len(labels)))
axes[2].set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
axes[2].set_yticklabels(labels, fontsize=12)

# Add black text annotations for difference
for i in range(conf_diff.shape[0]):
    for j in range(conf_diff.shape[1]):
        axes[2].text(j, i, f'{conf_diff[i, j]:.0f}', ha='center', va='center', color='black', fontsize=10)

plt.tight_layout()
#plt.savefig("genus_confusion_comparison_vertical.pdf", dpi=300, bbox_inches='tight')
plt.show()




#######################################################################################
#for species level
###############################################################################



