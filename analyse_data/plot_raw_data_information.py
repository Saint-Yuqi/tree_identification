import matplotlib.pyplot as plt
import pandas as pd

file_12 = "tree_statistics_12.csv"
file_34 = "tree_statistics_34.csv"

df_12 = pd.read_csv(file_12)
df_34 = pd.read_csv(file_34)

nr = 12

def create_plot(df,nr, x_input, y_input):

    plt.figure(figsize=(10,6))
    plt.bar(df[x_input], df[y_input])
    plt.xlabel(x_input)
    plt.ylabel(y_input)
    plt.title(f"File {nr}, {y_input}" )
    plt.xticks(rotation=45, ha="right")  # rotate labels if they overlap
    plt.tight_layout()  # adjust layout to fit labels
    plt.yscale('log')
    plt.show()
    plt.savefig(f"Speciedistribution_{nr}_{y_input}.png")
print("done")


create_plot(df_12, 12,"Tree Name", "train_rel_im_pres")
create_plot(df_34, 34, "Tree Name", "train_rel_im_pres")
