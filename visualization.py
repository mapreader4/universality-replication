import csv
import math

import matplotlib.pyplot as plt
import numpy as np

import mlp_mod_arithm as mma

def bar_plot_representations(representations, data):
    width = 0.2
    positions1 = np.arange(len(representations)).astype(float)
    positions2 = [x + width for x in positions1]
    positions3 = [x + width for x in positions2]
    positions4 = [x + width for x in positions3]

    plt.figure(figsize=(10, 6))
    plt.bar(positions1, data[1].copy(), width=width, label='Model 1')
    plt.bar(positions2, data[2], width=width, label='Model 2')
    plt.bar(positions3, data[3], width=width, label='Model 3')
    plt.bar(positions4, data[4], width=width, label='Model 4')

    plt.xlabel('Representations')
    plt.ylabel(data["name"])
    plt.ylim(bottom=0)
    plt.legend()

    plt.savefig(f"{data["saveto"]}.png")
    plt.close()

models = list(range(1,5))
representations = list(range(math.ceil(mma.MODULUS/2)))

logits = {"name": "Logit Similarity", "saveto": "logits"}
lefts = {"name": "%% of Left Embedding Explained", "saveto": "lefts"}
rights = {"name": "%% of Right Embedding Explained", "saveto": "rights"}
outs = {"name": "%% of Unembedding Explained", "saveto": "outs"}

filename = "model_representations.csv"
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)

    for model_num in models:
        logits[model_num] = []
        lefts[model_num] = []
        rights[model_num] = []
        outs[model_num] = []
        for representation_num in representations:
            line = next(csvreader)
            next(csvreader)
            logits[model_num].append(float(line[2]))
            lefts[model_num].append(float(line[3]))
            rights[model_num].append(float(line[4]))
            outs[model_num].append(float(line[5]))

bar_plot_representations(representations, logits)
bar_plot_representations(representations, lefts)
bar_plot_representations(representations, rights)
bar_plot_representations(representations, outs)