import csv
import math
import random

import torch
from torch import nn
from tqdm import tqdm

import mlp_mod_arithm as mma

def build_full_dataset():
    addition_table = []
    for i in range(mma.MODULUS):
        for j in range(mma.MODULUS):
            onehot_i = mma.number_to_onehot(i)
            onehot_j = mma.number_to_onehot(j)
            addition_table.append((onehot_i, onehot_j))

    left_list, right_list = zip(*addition_table)

    left = torch.stack(left_list)
    right = torch.stack(right_list)

    return (left, right)

def logit_similarity(k, actual_logits):
    predicted_logits = torch.arange(mma.MODULUS).view(mma.MODULUS, 1, 1) + torch.arange(mma.MODULUS).view(1, mma.MODULUS, 1) - torch.arange(mma.MODULUS).view(1, 1, mma.MODULUS)
    predicted_logits = 2 * torch.cos(2 * torch.pi * k * predicted_logits / mma.MODULUS)
    predicted_logits = predicted_logits.flatten()
    return nn.functional.cosine_similarity(actual_logits, predicted_logits, dim=-1)

def make_rep_space(k):
    if k == 0:
        return torch.ones(mma.MODULUS).view(mma.MODULUS, 1)
    else:
        cosines = torch.cos(2 * torch.pi * k * torch.arange(mma.MODULUS) / mma.MODULUS).view(mma.MODULUS, 1)
        sines = torch.sin(2 * torch.pi * k * torch.arange(mma.MODULUS) / mma.MODULUS).view(mma.MODULUS, 1)
        return torch.cat((cosines, sines), dim=1)

def rep_space_debug(k):
    representation = make_rep_space(k)
    Q, R = torch.linalg.qr(representation)
    return Q

def rep_space_projection(k, embedding):
    representation = make_rep_space(k)
    Q, R = torch.linalg.qr(representation)
    projection = torch.matmul(Q.T, embedding)
    return (torch.linalg.matrix_norm(projection) ** 2) / (torch.linalg.matrix_norm(embedding) ** 2)
            
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = mma.MLP(mma.MODULUS)
model.load_state_dict(torch.load('model_weights.pth', map_location=device))
model.eval()

left_embedding = model.left_embedding.weight
right_embedding = model.right_embedding.weight
mlp = model.mlp[0].weight
unembedding = model.unembedding.weight

left_input, right_input = build_full_dataset()
actual_logits = model(left_input, right_input)
actual_logits = actual_logits.flatten()

for k in range(0, math.ceil(mma.MODULUS/2)):
    if ((logit_sim := logit_similarity(k, actual_logits)) > 0.005):
        print(f'Representation {k} has logit similarity {logit_sim}')
    if ((left_sim := rep_space_projection(k, left_embedding.T)) > 0.05):
        print(f'Representation {k} explains {left_sim:.2%} of left embedding')
    if ((right_sim := rep_space_projection(k, right_embedding.T)) > 0.05):
        print(f'Representation {k} explains {right_sim:.2%} of right embedding')
    if ((out_sim := rep_space_projection(k, unembedding)) > 0.05):
        print(f'Representation {k} explains {out_sim:.2%} of unembedding')