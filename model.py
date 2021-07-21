#!/usr/bin/env python3

import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

class HeteroRGCNLayer(nn.Module):
    """Graph convolutional layer for relational data that supports
    multiple edge types.

    Parameters
    ----------
    in_size : int
        Number of input features for the graph convolutional layer.
    out_size : int
        Number of output features for the graph convolutional layer.
    etypes : list of str
        List of strings representing the names of all edge types in the graph.
    """
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()

        self.weight = nn.ModuleDict({
            name: nn.Linear(in_size, out_size) for name in etypes
        })

    def forward(self, G, feat_dict):
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            Wh = self.weight[etype](feat_dict[srctype])

            G.nodes[srctype].data['Wh_%s' % etype] = Wh

            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))

        G.multi_update_all(funcs, 'sum')

        return { ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes }


class HeteroRGCNNCModel(nn.Module):
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
        super(HeteroRGCNNCModel, self).__init__()

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
    # Margin loss:
    n_edges = pos_score.shape[0]
    return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()

class HeteroRGCNEPModel(nn.Module):
    """Relational GCN for heterogeneous graphs designed to predict
    missing edges between chemicals and Tox21 assays.
    """
    def __init__(self, G):
        super(HeteroRGCNEPModel, self).__init__()

    def forward(self, G, h, etype):
        with G.local_scope():
            G.ndata['h'] = h
            G.apply_edges(F.u_dot_v('h', 'h', 'score'), etype=etype)
            return G.edges[etype].data['score']