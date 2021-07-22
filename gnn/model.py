#!/usr/bin/env python3

import dgl.function as fn
from numpy import e
import torch
import torch.nn as nn
import torch.nn.functional as F

import ipdb

class HeteroRGCNLayer(nn.Module):
    """Graph convolutional layer for relational data that supports
    multiple edge types.

    Parameters
    ----------
    in_size_dict : int
        A dictionary mapping edge types to their input size. For any given edge
        type, the input size is equal to the dimensionality of the source
        node's feature matrix.
    out_size : int
        Number of output features for the graph convolutional layer.
    etypes : list of str
        List of strings representing the names of all edge types in the graph.
    """
    def __init__(self, in_size_dict, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()

        self.weight = nn.ModuleDict({
            name: nn.Linear(in_size_dict[name], out_size).to('cuda:0') for name in etypes
        })

    def forward(self, G, feat_dict):
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            Wh = self.weight[etype](feat_dict[srctype]).to('cuda:0')

            G.nodes[srctype].data['Wh_%s' % etype] = Wh

            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))

        G.multi_update_all(funcs, 'sum')

        return { ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes }


class NCModel(nn.Module):
    """Relational Convolutional Graph NN for heterogeneous graphs.

    The neural network consists of two HeteroRGCNLayers stacked in an
    encoder/decoder arrangement.

    Parameters
    ----------
    G : dgl.heterograph.DGLHeteroGraph
        Heterogeneous graph on which to perform a learning task.
    in_size : int
        Number of input node features. In this implementation all nodes have the same
        number of features.
    hidden_size : int
        Number of features in the hidden layer. Note that this implementation only
        contains a single hidden layer.
    out_size : int
        Number of output node features. If you don't have a reason to do otherwise, this
        should probably be the same as `in_size`.
    """
    def __init__(self, G, in_size, hidden_size, out_size):
        super(NCModel, self).__init__()

        embed_dict = {
            ntype: nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), in_size)) for ntype in G.ntypes
        }

        for key, embed in embed_dict.items():
            nn.init.xavier_uniform_(embed)

        self.embed = nn.ParameterDict(embed_dict)

        self.layer1 = HeteroRGCNLayer(in_size, hidden_size, G.etypes)
        self.layer2 = HeteroRGCNLayer(hidden_size, out_size, G.etypes)

    def forward(self, G):
        h_dict = self.layer1(G, self.embed)
        h_dict = { k : F.leaky_relu(h) for k, h in h_dict.items() }
        h_dict = self.layer2(G, h_dict)

        return h_dict['chemical']

def compute_ep_loss(pos_score, neg_score):
    """Compute the Margin Loss between positive and negative edges for Edge
    Prediction.
    
    Parameters
    ----------
    pos_score : torch.tensor
    neg_score : torch.tensor
    """
    # Margin loss:
    n_edges = pos_score.shape[0]
    return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()

class HeteroDotProductPredictor(nn.Module):
    def forward(self, G, h, etype):
        with G.local_scope():
            G.ndata['h'] = h
            G.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return G.edges[etype].data['score']

class EPModel(nn.Module):
    """Relational GCN for heterogeneous graphs designed to predict
    missing edges between chemicals and Tox21 assays.

    Parameters
    ----------
    in_size_dict : dict
        A dictionary mapping edge types to their input size. For any given edge
        type, the input size is equal to the dimensionality of the source
        node's feature matrix.
    hidden_size : int
        The dimensionality of the hidden (encoded) representation of nodes.
        Currently, this is a single, fixed dimensionality, but it may be
        extended to allow for different hidden sizes for each node type.
    out_size : int
        The dimensionality of nodes' output representations.
    rel_names : list of str
        List of all edge type names in the graph. These should be the same as
        the edge types described in `in_size_dict` (consider removing to lessen
        redundancy).
    """
    def __init__(self, in_size_dict, hidden_size, out_size, rel_names):
        super(EPModel, self).__init__()

        # NOTE: May need to modify to take _one_ rel name
        self.sage = HeteroRGCNLayer(in_size_dict, out_size, rel_names)

        self.pred = HeteroDotProductPredictor()

    def forward(self, g, neg_g, x, etype):
        # An 'encoder', of sorts
        h = self.sage(g, x)

        return self.pred(g, h, etype), self.pred(neg_g, h, etype)