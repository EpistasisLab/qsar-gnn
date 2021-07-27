#!/usr/bin/env python3

import argparse
import dgl
import numpy as np
import pandas as pd
import pickle as pkl
import torch
import torch.nn.functional as F

import ipdb

from gnn.link_prediction import LinkPredictor, compute_lp_loss
from gnn.node_classification import NodeClassifier

if torch.cuda.is_available():
    DEVICE = 'cuda:0'
else:
    DEVICE = 'cpu'


def preprocess_edges(graph):
    chem = pd.read_csv("./data/chemicals.csv")
    maccs = torch.tensor([[int(x) for x in xx] for xx in chem.maccs]).float().to(DEVICE)
    node_features = {
        'chemical': maccs,
        #'chemical': torch.ones((graph.number_of_nodes(ntype='chemical'))).unsqueeze(1).to(DEVICE),
        'assay': torch.ones((graph.number_of_nodes(ntype='assay'))).unsqueeze(1).to(DEVICE),
        'gene': torch.ones((graph.number_of_nodes(ntype='gene'))).unsqueeze(1).to(DEVICE)
    }
    input_type_map = dict([(x[1], x[0]) for x in graph.canonical_etypes])
    node_sizes = { k: v.shape[1] for k, v in node_features.items() }
    edge_input_sizes = { k: node_features[v].shape[1] for k, v in input_type_map.items() }

    return node_features, node_sizes, edge_input_sizes

def drop_node_return_binary_labels(graph, ntype, node_index, pos_etype, neg_etype):
    """Drop a node from the graph and save its original connectivity as labels
    for supervised learning (especially node classification).

    Parameters
    ----------
    graph : dgl.DGLGraph
        Graph on which to run the operation.
    ntype : str
        Node type of the node to drop from the graph.
    node_index : int
        Integer index of the (typed) node to drop.
    pos_etype : str
        Name of an incoming edge, the source of which will be labeled 1.
    neg_etype : str
        Name of an incoming edge, the source of which will be labeled 0.

    Notes
    -----
    This is not an 'in place' operation. If you want to overwrite the original
    graph, you should reassign to it when calling the function.
    """
    # Get node connectivity and build label indices
    # FIX: Need to get the source node ids only!
    
    pos_nodes = graph.in_edges(node_index, form='eid', etype=pos_etype)
    neg_nodes = graph.in_edges(node_index, form='eid', etype=neg_etype)

    # Remove node
    new_graph = dgl.remove_nodes(graph, node_index, ntype=ntype)

    # Return both the transformed graph and the labels
    return new_graph, pos_nodes, neg_nodes

def collate_labels(graph, ntype, pos_idxs, neg_idxs, ratio=(0.8, 0.1, 0.1)):
    """Make training/validation/testing sets along with labels.
    """
    assert sum(ratio) == 1.0
    
    all_node_idxs = graph.nodes(ntype)

    pos_tensor = torch.cat((pos_idxs.unsqueeze(0), torch.ones_like(pos_idxs).unsqueeze(0)))
    neg_tensor = torch.cat((neg_idxs.unsqueeze(0), torch.zeros_like(neg_idxs).unsqueeze(0)))
    labeled_nodes = torch.cat((pos_tensor, neg_tensor), dim=1)
    
    idxs = torch.randperm(labeled_nodes.shape[1])
    shuffled = labeled_nodes[:,idxs]

    split_points = (
        round(shuffled.shape[1] * ratio[0]),
        round(shuffled.shape[1] * (ratio[0]+ratio[1]))
    )

    train = shuffled[:, :split_points[0]]
    val = shuffled[:, split_points[0]:split_points[1]]
    test = shuffled[:, split_points[1]:]

    return train, val, test
    
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


def link_prediction(args):
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

    k = 5
    
    node_features, node_sizes, edge_input_sizes = preprocess_edges(G)
    
    ep_model = EPModel(edge_input_sizes, 20, 5, G.etypes)
    opt = torch.optim.Adam(ep_model.parameters())

    for epoch in range(100):
        neg_G = construct_negative_graph(G, k, ('chemical', 'chemicalhasactiveassay', 'assay'))
        pos_score, neg_score = ep_model(G.to(DEVICE), neg_G.to(DEVICE), node_features, ('chemical', 'chemicalhasactiveassay', 'assay'))
        
        # margin loss
        loss = compute_ep_loss(pos_score, neg_score)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        print("epoch: %3d; margin loss: %.5f" % (epoch, loss.item()))

        # Now, we need to figure out something to do with the trained model!

    ipdb.set_trace()

def node_classification(args, label_assay_idx):
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
    label_assay_idx : int
        Index of 
    """
    G = dgl.load_graphs(args.graph_file)[0][0]


    _ = """
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
    """

    # Remove the prediction task node and get labels before doing anything else
    G, pos_nodes, neg_nodes = drop_node_return_binary_labels(G, 'assay', label_assay_idx, 'chemicalhasactiveassay', 'chemicalhasinactiveassay')

    train, val, test = collate_labels(G, 'chemical', pos_nodes, neg_nodes)

    train_idx = train[0,:]
    train_labels = train[1,:]
    val_idx = val[0,:]
    val_labels = val[1,:]
    test_idx = test[0,:]
    test_labels = test[1,:]

    # Note: We don't do anything with the node features (yet)
    node_features, node_sizes, edge_input_sizes = preprocess_edges(G)

    model = NodeClassifier(G, node_sizes, edge_input_sizes, 2, 3)  # (graph, input_size, output_size, num_edge_types)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_val_acc = 0
    best_test_acc = 0

    for epoch in range(500):
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
            try:
                print('Epoch %4d: Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f)' % (
                    epoch,
                    loss.item(),
                    train_acc.item(),
                    val_acc.item(),
                    best_val_acc.item(),
                    test_acc.item(),
                    best_test_acc.item(),
                ))
            except AttributeError:
                ipdb.set_trace()
                print()
    
def main(args):
    if args.task == "lp":
        link_prediction(args)
    elif args.task == "nc":
        node_classification(args, 1)

if __name__=="__main__":
    tasks = ['nc', 'lp']
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
    main(args)