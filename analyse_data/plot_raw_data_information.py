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
    plt.xlabel("class name")
    plt.ylabel(y_input)
    plt.title(f"SemSegm {nr}, {y_input}" )
    plt.xticks(rotation=45, ha="right")  # rotate labels if they overlap
    plt.tight_layout()  # adjust layout to fit labels
    plt.yscale('log')
    plt.show()
    plt.savefig(f"Speciedistribution_{nr}_{y_input}.png")
print("done")


create_plot(df_12, 12,"Tree Name", "train_rel_pix_pres")
create_plot(df_34, 34, "Tree Name", "train_rel_pix_pres")
create_plot(df_12, 12,"Tree Name", "test_rel_pix_pres")
create_plot(df_34, 34, "Tree Name", "test_rel_pix_pres")
create_plot(df_12, 12,"Tree Name", "val_rel_pix_pres")
create_plot(df_34, 34, "Tree Name", "val_rel_pix_pres")




def create_combined_plot(df, nr, x_input, y_inputs):
    plt.figure(figsize=(10,6))

    colors = ['tab:blue', 'tab:orange', 'tab:green']  # train, test, val
    alphas = [0.7, 0.5, 0.3]  # transparency levels

    for y_input, color, alpha in zip(y_inputs, colors, alphas):
        plt.bar(df[x_input], df[y_input], color=color, alpha=alpha, label=y_input)

    plt.xlabel("Class name")
    plt.ylabel("Relative pixel presence")
    plt.title(f"SemSegm {nr}: Train / Test / Val Relative Pixel Presence")
    plt.xticks(rotation=45, ha="right")
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"Speciedistribution_{nr}_combined.png")
    plt.show()

# Example calls
create_combined_plot(df_12, 12, "Tree Name", ["train_rel_pix_pres", "test_rel_pix_pres", "val_rel_pix_pres"])
create_combined_plot(df_34, 34, "Tree Name", ["train_rel_pix_pres", "test_rel_pix_pres", "val_rel_pix_pres"])


