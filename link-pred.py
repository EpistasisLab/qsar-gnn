#!/usr/bin/env python3

import dgl
from dgl import load_graphs
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import pickle as pkl

import ipdb

# Loading example dataset:
import scipy.io
import urllib.request

# Local import(s):
from model import HeteroRGCN

# LOAD THE REAL DATA:
G = load_graphs("./data/graph.bin")[0][0]
print(G)

# Get labels for a single prediction task:
active_assays = pkl.load(open("./data/active_assays.pkl", 'rb')).numpy()
active_labels = np.ones(len(active_assays), dtype=int)
active = np.vstack([active_assays, active_labels])  # First row - indices; second row - labels
inactive_assays = pkl.load(open("./data/inactive_assays.pkl", 'rb')).numpy()
inactive_labels = np.zeros(len(inactive_assays), dtype=int)
inactive = np.vstack([inactive_assays, inactive_labels])

idx_labels_merged = np.hstack([active, inactive])  # First row - indices; second row - labels
# Now, we shuffle them
idx_labels_merged = idx_labels_merged[:, np.random.permutation(idx_labels_merged.shape[1])]

train_idx = torch.tensor(idx_labels_merged[0,:5000].squeeze()).long()
train_labels = torch.tensor(idx_labels_merged[1,:5000].squeeze()).long()
val_idx = torch.tensor(idx_labels_merged[0,5000:6000].squeeze()).long()
val_labels = torch.tensor(idx_labels_merged[1,5000:6000].squeeze()).long()
test_idx = torch.tensor(idx_labels_merged[0,6000:].squeeze()).long()
test_labels = torch.tensor(idx_labels_merged[1,6000:].squeeze()).long()


model = HeteroRGCN(G, 2, 2, 3)  # (graph, input_size, output_size, num_edge_types)
opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

best_val_acc = 0
best_test_acc = 0

for epoch in range(1000):
    logits = model(G)

    try:
        loss = F.cross_entropy(logits[train_idx], train_labels)
    except TypeError:
        ipdb.set_trace()
        print()

    pred = logits.argmax(1)

    train_acc = (pred[train_idx] == train_labels).float().mean()
    val_acc = (pred[val_idx] == val_labels).float().mean()
    test_acc = (pred[test_idx] == test_labels).float().mean()

    if best_val_acc < val_acc:
        best_val_acc = val_acc
    if best_test_acc < test_acc:
        best_test_acc = test_acc

    opt.zero_grad()
    loss.backward()
    opt.step()

    if epoch % 5 == 0:
        print('Epoch %4d: Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f)' % (
            epoch,
            loss.item(),
            train_acc.item(),
            val_acc.item(),
            best_val_acc.item(),
            test_acc.item(),
            best_test_acc.item(),
        ))

ipdb.set_trace()