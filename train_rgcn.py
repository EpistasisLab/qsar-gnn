#!/usr/bin/env python3

import argparse
import dgl
import numpy as np
import pickle as pkl
import torch
import torch.nn.functional as F

import ipdb

from model import HeteroRGCNNCModel, HeteroRGCNEPModel, compute_ep_loss


def construct_negative_graph(graph, k, etype):
    """Construct a negative graph for negative sampling in edge prediction.
    
    This implementation is designed for heterogeneous graphs - the user specifies
    the edge type on which negative sampling will be performed.

    Parameters
    ----------
    graph : dgl.heterograph.DGLHeterograph
        Graph on which the sampling will be performed.
    k : int
        Number of negative examples to retrieve.
    etype : tuple
        A tuple in the format (subj_node_type, edge_label, obj_node_type) corresponding
        to the edge on which negative sampling will be performed.
    """
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,))
    
    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict= { ntype: graph.num_nodes(ntype) for ntype in graph.ntypes }
    )


def edge_prediction(args):
    """Predict edges in a heterogeneous graph given a particular edge type.

    For this implementation, the edge type we are predicting is:
    `('chemical', 'chemicalhasactiveassay', 'assay')`

    There are two approaches for training the network:
    1. Train known edges against a negative sampling of the entire graph, using
       margin loss (or equivalent) to maximize the difference between known edges
       and the background "noise distribution" of randomly sampled edges.
    2. Use a predetermined edge (e.g., `'chemicalhasinactiveassay'`) instead as the
       negative graph. This approach may be more powerful. Cross-entropy loss also
       may be more appropriate than margin loss in this scenario.

    Parameters
    ----------
    args : (namespace output of argparse.parse_args() - see below for details)
    """
    G = dgl.load_graphs(args.graph_file)[0][0]

    #ipdb.set_trace()

    # Do some more processing on the graph here
    # TODO

    k = 5
    ep_model = HeteroRGCNEPModel(10, 20, 5, G.etypes)
    node_features = {

    }
    opt = torch.optim.Adam(ep_model.parameters())

    for epoch in range(100):
        neg_G = construct_negative_graph(G, k, ('chemical', 'chemicalhasactiveassay', 'assay'))
        pos_score, neg_score = ep_model(G, neg_G, node_features, ('chemical', 'chemicalhasactiveassay', 'assay'))
        
        # margin loss
        loss = compute_ep_loss(pos_score, neg_score)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        print(loss.item())
    

def node_classification(args):
    """Predict node labels in a heterogeneous graph given a particular node type
    and a dataset of known node labels.

    Here, the node type is `'chemical'` and the labels we predict are annotations
    representing edge connectivity to a (deleted) `'assay'` node where 0 is
    `'chemicalhasinactiveassay'` edges and 1 is `'chemicalhasactiveassay'` edges.
    Labels are not provided for nodes with unknown activity annotations to the
    assay of interest.

    Parameters
    ----------
    args : (namespace output of argparse.parse_args() - see below for details)
    """
    G = dgl.load_graphs(args.graph_file)[0][0]

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

    model = HeteroRGCNNCModel(G, 2, 2, 3)  # (graph, input_size, output_size, num_edge_types)
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
    
def main(args):
    if args.task == "ep":
        edge_prediction(args)
    elif args.task == "nc":
        node_classification(args)

if __name__=="__main__":
    tasks = ['nc', 'ep']
    parser = argparse.ArgumentParser(description="Train a heterogeneous RGCN on a prediction task.")
    parser.add_argument("--task", type=str, default="nc",
                        help="Type of prediction task to perform.",
                        choices=tasks)
    parser.add_argument("--graph-file", type=str, default="./data/graph.bin",
                        help="File location where the DGL heterograph is stored.")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate for the NN optimizer.")

    parser.set_defaults(validation=True)

    args = parser.parse_args()
    print(args)
    main(args)