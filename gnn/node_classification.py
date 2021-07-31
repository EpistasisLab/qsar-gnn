import torch
from torch import nn
import torch.nn.functional as F
import dgl.nn as dglnn

from .model import HeteroRGCNLayer

DEVICE = 'cpu'

class NodeClassifier(nn.Module):
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
    def __init__(self, G, node_sizes, edge_input_sizes, hidden_size, out_size):
        super(NodeClassifier, self).__init__()

        # Node embeddings. Note that there is a different embedding dict for each node type.
        embed_dict = {
            ntype: nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), node_sizes[ntype])) for ntype in G.ntypes
        }

        # Initialize the embedding matrices as xavier uniform
        for key, embed in embed_dict.items():
            nn.init.xavier_uniform_(embed)

        self.embed = nn.ParameterDict(embed_dict)

        self.layer1 = HeteroRGCNLayer(edge_input_sizes, hidden_size, G.etypes)
        self.layer2 = HeteroRGCNLayer(hidden_size, out_size, G.etypes)

    def forward(self, G):
        h_dict = self.layer1(G, self.embed)
        h_dict = { k : F.leaky_relu(h) for k, h in h_dict.items() }
        h_dict = self.layer2(G, h_dict)
        
        return h_dict['chemical']


class NodeClassifierConv(nn.Module):
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
    def __init__(self, G, node_sizes, edge_input_sizes, hidden_size, out_size):
        super(NodeClassifierConv, self).__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(rel_size, hidden_size) for rel, rel_size in edge_input_sizes.items()
        }, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hidden_size, out_size) for rel, _ in edge_input_sizes.items()
        }, aggregate='sum')

    def forward(self, G, inputs):
        h_dict = self.conv1(G, inputs)
        h_dict = { k : F.leaky_relu(h) for k, h in h_dict.items() }
        h_dict = self.conv2(G, h_dict)

        return h_dict['chemical']