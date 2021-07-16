#!/usr/bin/env python3

import dgl
import dgl.function as fn
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import ipdb

g = dgl.load_graphs('E:\\data\\comptoxai\\psb\\graph.bin')[0][0]

def construct_negative_graph(graph, k, etype):
  utype, _, vtype = etype
  src, dst = graph.edges(etype=etype)
  neg_src = src.repeat_interleave(k)
  neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,))

class HeteroDotProductPredictor(nn.Module):
  def forward(self, graph, h, etype):
    with graph.local_scope():
      graph.ndata['h'] = h
      graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
      return graph.edges[etype].data['score']

class RGCN(nn.Module):
  """A Relational Graph Convolutional Network module, suitable for performing
  inference on heterogeneous graphs.

  Parameters
  ----------
  in_feats : int
    Dimensionality of the input features.
  hid_feats : int
    Dimensionality of the hidden features.
  out_feats : int
    Dimensionality of the output features.
  rel_names : list of str
    List of strings that represent edge labels in the graph.
  """
  def __init__(self, in_feats, hid_feats, out_feats, rel_names):
    super().__init__()

    self.conv1 = dglnn.HeteroGraphConv(
      { rel: dglnn.GraphConv(in_feats, hid_feats) for rel in rel_names },
      aggregate='sum'
    )
    self.conv2 = dglnn.HeteroGraphConv(
      { rel: dglnn.GraphConv(hid_feats, out_feats) for rel in rel_names },
      aggregate='sum'
    )

  def forward(self, graph, inputs):
    """Inputs are node features."""
    ipdb.set_trace()
    h = self.conv1(graph, inputs)
    h = { k: F.relu(v) for k, v in h.items() }
    h = self.conv2(graph, h)
    return h

class Model(nn.Module):
  def __init__(self, in_features, hidden_features, out_features, rel_names):
    super().__init__()
    self.sage = RGCN(in_features, hidden_features, out_features, rel_names)
    self.pred = HeteroDotProductPredictor()

  def forward(self, g, neg_g, x, etype):
    h = self.sage(g, x)
    return self.pred(g, h, etype), self.pred(neg_g, h, etype)

def compute_loss(pos_score, neg_score):
  # Margin loss (Maybe refactor to cross entropy?)
  n_edges = pos_score.shape[0]
  return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()


k = 5

model = Model(10, 20, 5, g.etypes)

# Define node features
chem_feats = g.nodes['chemical'].data['maccs']
gene_feats = g.nodes['gene'].data['dummy']
assay_feats = g.nodes['assay'].data['dummy']
node_features = {
  'chemical': chem_feats,
  'gene': gene_feats,
  'assay': assay_feats
}

opt = Adam(model.parameters())
for epoch in range(10):
  neg_g = construct_negative_graph(g, k, ('chemical', 'chemicalhasactiveassay', 'assay'))
  pos_score, neg_score = model(g, neg_g, node_features, ('chemical', 'chemicalhasactiveassay', 'assay'))
  loss = compute_loss(pos_score, neg_score)
  opt.zero_grad()
  loss.backward()
  opt.step()
  print(loss.item())