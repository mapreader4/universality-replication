import csv
import math
import random

import torch
from torch import nn
from tqdm import tqdm

#hyperparameters of model (fixed)
LR_EMBEDDING_SIZE = 256
INPUT_MLP_SIZE = 2 * LR_EMBEDDING_SIZE
OUTPUT_MLP_SIZE = 128
LEARNING_RATE = 0.001
BETA_1 = 0.9
BETA_2 = 0.98
WEIGHT_DECAY = 1.0

#modulus to use for modular arithmetic
MODULUS = 113
DATA_SIZE = MODULUS * MODULUS
TRAIN_SIZE = math.ceil(0.4 * DATA_SIZE)
TEST_SIZE = DATA_SIZE - TRAIN_SIZE

#hyperparameters of model (adjustable)
EPOCH_COUNT = 250000
SAVE_LOSS_STEP = 100

#make sure model runs on gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#translate numbers to onehot vectors
def number_to_onehot(number):
    onehot = torch.zeros(MODULUS)
    onehot[number % MODULUS] = 1
    return onehot

#construct training and test datasets from subsets of all additions
def build_dataset():
    addition_table = []
    for i in range(MODULUS):
        for j in range(MODULUS):
            onehot_i = number_to_onehot(i)
            onehot_j = number_to_onehot(j)
            res = (i + j) % MODULUS
            addition_table.append((onehot_i, onehot_j, res))
    random.shuffle(addition_table)
    return (addition_table[:TRAIN_SIZE], addition_table[TRAIN_SIZE:])

#format data for model input
def format_data(dataset):
    left_list, right_list, labels_list = zip(*dataset)

    left = torch.stack(left_list)
    right = torch.stack(right_list)
    labels = torch.tensor(labels_list)

    return (left, right, labels)

#define one-layer MLP architecture
class MLP(nn.Module):
    
    def __init__(self, element_count):
        super().__init__()

        self.left_embedding = nn.Linear(element_count, LR_EMBEDDING_SIZE, bias=False)
        self.right_embedding = nn.Linear(element_count, LR_EMBEDDING_SIZE, bias=False)

        layers = []
        layers.append(nn.Linear(INPUT_MLP_SIZE, OUTPUT_MLP_SIZE, bias=False))
        layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)

        self.unembedding = nn.Linear(OUTPUT_MLP_SIZE, element_count, bias=False)

    def forward(self, a, b):
        left = self.left_embedding(a)
        right = self.right_embedding(b)
        embedded = torch.cat((left, right), dim=1)
        mlp_out = self.mlp(embedded)
        logits = self.unembedding(mlp_out)
        return logits

def main():
    #create and format data
    train_data, test_data = build_dataset()
    train_left, train_right, train_labels = format_data(train_data)
    test_left, test_right, test_labels = format_data(test_data)

    #move data to gpu
    train_left, train_right, train_labels = train_left.to(device), train_right.to(device), train_labels.to(device)
    test_left, test_right, test_labels = test_left.to(device), test_right.to(device), test_labels.to(device)

    #initialize model, loss function, and optimizer
    model = MLP(MODULUS).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(BETA_1, BETA_2), weight_decay=WEIGHT_DECAY)

    #training loop
    test_loss = ''
    for epoch in (pbar := tqdm(range(EPOCH_COUNT), desc="Loss Unevaluated")):
        #save loss every so often
        if (epoch % SAVE_LOSS_STEP) == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(test_left, test_right)
                accuracy = (torch.argmax(test_outputs, dim=1) == test_labels).float().sum() / len(test_labels)
                test_loss = loss_function(test_outputs, test_labels)
                pbar.set_description(f'Loss {test_loss}')
                with open('accuracy_and_loss.csv', 'a') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([epoch, accuracy.item(), test_loss.item()])

        #training step
        model.train()
        optimizer.zero_grad()
        outputs = model(train_left, train_right)
        loss = loss_function(outputs, train_labels)
        loss.backward()
        optimizer.step()

    #save model, save loss one last time
    torch.save(model.state_dict(), 'model_weights.pth')
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_left, test_right)
        accuracy = (torch.argmax(test_outputs, dim=1) == test_labels).float().sum() / len(test_labels)
        test_loss = loss_function(test_outputs, test_labels)
        with open('accuracy_and_loss.csv', 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([EPOCH_COUNT, accuracy.item(), test_loss.item()])

if __name__ == "__main__":
    main()