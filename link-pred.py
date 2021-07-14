#!/usr/bin/env python3

import dgl
from dgl import load_graphs
import numpy as np
import torch
import torch.nn.functional as F

# Loading example dataset:
import scipy.io
import urllib.request

# Local import(s):
from model import HeteroRGCN

# LOAD THE REAL DATA:
# G = load_graphs("./graph.bin")[0][0]
# G.edges

# LOAD THE EXAMPLE DATA:
data_url = 'https://data.dgl.ai/dataset/ACM.mat'
data_file_path = '/tmp/ACM.mat'
urllib.request.urlretrieve(data_url, data_file_path)
data = scipy.io.loadmat(data_file_path)
G = dgl.heterograph({
        ('paper', 'written-by', 'author') : data['PvsA'].nonzero(),
        ('author', 'writing', 'paper') : data['PvsA'].transpose().nonzero(),
        ('paper', 'citing', 'paper') : data['PvsP'].nonzero(),
        ('paper', 'cited', 'paper') : data['PvsP'].transpose().nonzero(),
        ('paper', 'is-about', 'subject') : data['PvsL'].nonzero(),
        ('subject', 'has', 'paper') : data['PvsL'].transpose().nonzero(),
    })
print(G)
pvc = data['PvsC'].tocsr()
# find all papers published in KDD, ICML, VLDB
c_selected = [0, 11, 13]  # KDD, ICML, VLDB
p_selected = pvc[:, c_selected].tocoo()
# generate labels
labels = pvc.indices
labels[labels == 11] = 1
labels[labels == 13] = 2
labels = torch.tensor(labels).long()
# generate train/val/test split
pid = p_selected.row
shuffle = np.random.permutation(pid)
train_idx = torch.tensor(shuffle[0:800]).long()
val_idx = torch.tensor(shuffle[800:900]).long()
test_idx = torch.tensor(shuffle[900:]).long()


model = HeteroRGCN(G, 10, 10, 3)  # (graph, input_size, output_size, num_edge_types)
opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

best_val_acc = 0
best_test_acc = 0

for epoch in range(300):
    logits = model(G)

    loss = F.cross_entropy(logits[train_idx], labels[train_idx])

    pred = logits.argmax(1)
    train_acc = (pred[train_idx] == labels[train_idx]).float().mean()
    val_acc = (pred[val_idx] == labels[val_idx]).float().mean()
    test_acc = (pred[test_idx] == labels[test_idx]).float().mean()

    if best_val_acc < val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc

    opt.zero_grad()
    loss.backward()
    opt.step()

    if epoch % 5 == 0:
        print('Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f)' % (
            loss.item(),
            train_acc.item(),
            val_acc.item(),
            best_val_acc.item(),
            test_acc.item(),
            best_test_acc.item(),
        ))
