import matplotlib.pyplot as plt
import numpy as np

alpha_ablation_x = [0, .2, .4, .6, .8, 1., 1.2, 1.4, 1.6, 1.8, 2.]
alpha_ablation_y = [
    ([50.9, 52.2, 52.6, 54.1, 53.6, 54.7, 53.6, 53.2, 53.0, 53.7, 52.8], "all"),
    ([64.7, 65.3, 62.9, 60.7, 58.0, 55.5, 54.0, 53.2, 52.1, 50.3, 49.8], "many"),
    ([53.1, 53.4, 54.7, 56.9, 59.8, 59.4, 60.7, 60.2, 59.8, 60.9, 61.6], "med"),
    ([33.1, 36.2, 38.4, 43.5, 41.2, 48.4, 45.3, 45.5, 46.7, 49.5, 46.6], "few"),
]

tau_ablation_x = [.05, .1, .15, .2, .25, .3, .35, .4, .45 , .5]
tau_ablation_y = [
    ([79.6, 80.3, 80.0, 80.1, 79.7, 79.6, 79.4, 79.3, 79.4, 79.7], "all"),
    ([84.4, 85.1, 84.6, 85.3, 84.5, 85.5, 84.4, 84.5, 84.7, 84.8], "many"),
    ([82.5, 81.4, 81.4, 80.7, 80.9, 81.0, 80.8, 80.9, 81.0, 80.7], "med"),
    ([70.9, 73.6, 73.2, 73.5, 72.9, 71.6, 72.1, 71.7, 71.6, 72.8], "few"),
]

def create_graph(title, x_label, y_label, x, y):
    plt.clf()
    plt.cla()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    for line in y:
        plt.plot(x, line[0], label=line[1], linestyle="--" if line[1] != "all" else "-")
    plt.legend()
    plt.savefig(f"plots/{x_label}.png")

create_graph("Effect on Different Values for Alpha", "Alpha", "Top-1 Accuracy", alpha_ablation_x, alpha_ablation_y)
create_graph("Effect on Different Values for Tau", "Tau", "Top-1 Accuracy", tau_ablation_x, tau_ablation_y)
