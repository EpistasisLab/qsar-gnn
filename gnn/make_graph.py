#!/usr/bin/env python3

# Export a graph for link prediction in an RGCN model via DGL/PyTorch
# from comptox_ai.db.graph_db import GraphDB
# db = GraphDB()
# # Get nodes
# db.run_cypher("""
# WITH "MATCH (n:Chemical) WHERE n.maccs IS NOT NULL RETURN id(n) AS node, n.maccs AS maccs;" AS chemicalsquery
# CALL apoc.export.csv.query(chemicalsquery, "file:/E:/data/comptoxai/psb/chemicals.csv", {})
# YIELD file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data
# RETURN file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data;
# """)
# db.run_cypher("""
# WITH "MATCH (n:Assay) RETURN id(n) AS node;" AS assaysquery
# CALL apoc.export.csv.query(assaysquery, "file:/E:/data/comptoxai/psb/assays.csv", {})
# YIELD file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data
# RETURN file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data;
# """)
# db.run_cypher("""
# WITH "MATCH (n:Gene) RETURN id(n) AS node;" AS genesquery
# CALL apoc.export.csv.query(genesquery, "file:/E:/data/comptoxai/psb/genes.csv", {})
# YIELD file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data
# RETURN file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data;
# """)
# # Get relationships
# db.run_cypher("""
# WITH "MATCH (n:Chemical)-[r]-(m:Assay) WHERE n.maccs IS NOT NULL RETURN id(n) AS node1, TYPE(r) AS edge, id(m) AS node2;" AS edgeqry
# CALL apoc.export.csv.query(edgeqry, "file:/E:/data/comptoxai/psb/chemical-assay.csv", {})
# YIELD file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data
# RETURN file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data;
# """)
# db.run_cypher("""
# WITH "MATCH (n:Chemical)-[r]-(m:Gene) WHERE n.maccs IS NOT NULL RETURN id(n) AS node1, TYPE(r) AS edge, id(m) AS node2;" AS edgeqry
# CALL apoc.export.csv.query(edgeqry, "file:/E:/data/comptoxai/psb/chemical-gene.csv", {})
# YIELD file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data
# RETURN file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data;
# """)
# db.run_cypher("""
# WITH "MATCH (n:Gene)-[r]-(m:Gene) RETURN id(n) AS node1, TYPE(r) AS edge, id(m) AS node2;" AS edgeqry
# CALL apoc.export.csv.query(edgeqry, "file:/E:/data/comptoxai/psb/gene-gene.csv", {})
# YIELD file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data
# RETURN file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data;
# """)

from numpy.core.fromnumeric import size
import torch
from dgl import save_graphs, load_graphs, heterograph, edge_type_subgraph
from dgl.data import DGLDataset
import pandas as pd
import numpy as np
import scipy.sparse
import ipdb
from os import path

import pickle as pkl

from torch.nn.functional import adaptive_avg_pool2d

class QSARDataset(DGLDataset):
  def __init__(self, name, rebuild=False):
    self.rebuild = rebuild
    super(QSARDataset, self).__init__(name)
  
  def has_cache(self):
    print("Checking if has cache...")
    if self.rebuild:
      return False
    
    if path.exists("./data/graph.bin"):
      # return True
      return False
    else:
      return False

  def download(self):
    print("Downloading: No need to download - local files only")
    pass  # We only use local files

  def process(self):
    print("Processing data...")

    self.read_source_files()
    self.parse_node_features()
    self.process_node_labels()  # For link prediction
    self.build_adjacency_matrices()
    self.make_heterograph()

  def read_source_files(self):
    print("  ...reading source files.")
    # Load node source files
    self.chemicals = pd.read_csv("../data/chemicals.csv")
    self.genes = pd.read_csv("../data/genes.csv")
    self.assays = pd.read_csv("../data/assays.csv")
    # Load edge source files
    self.chemical_assay = pd.read_csv("../data/chemical-assay.csv")
    self.chemical_gene = pd.read_csv("../data/chemical-gene.csv")
    self.gene_gene = pd.read_csv("../data/gene-gene.csv")

  def parse_node_features(self):
    print("  ...parsing node features.")
    # Get MACCS data for chemicals
    maccs_ndarray = np.empty(shape=(len(self.chemicals), len(self.chemicals.maccs[0])))
    for i, m in enumerate(self.chemicals.maccs):
      maccs_ndarray[i,:] = [int(mm) for mm in m]
    self.maccs_tensor = torch.tensor(maccs_ndarray, dtype=torch.bool)

  def process_node_labels(self):
    print("  ...processing node labels.")
    inactive_assays = self.make_rel_t('chemical', 'CHEMICALHASINACTIVEASSAY', 'assay')
    active_assays = self.make_rel_t('chemical', 'CHEMICALHASACTIVEASSAY', 'assay')

    active_assay_mask = active_assays[1] == 47
    inactive_assay_mask = inactive_assays[1] == 47

    #ipdb.set_trace()
    
    active_assay_nodes = active_assays[0][active_assay_mask]
    inactive_assay_nodes = inactive_assays[0][inactive_assay_mask]
    pkl.dump(active_assay_nodes, open("../data/active_assays.pkl", 'wb'))
    pkl.dump(inactive_assay_nodes, open("../data/inactive_assays.pkl", 'wb'))
    
  def make_adjacency(self, s_node, rel, o_node):
    """
    Make an adjacency list for a specific metaedge.

    Parameters
    ----------
    s_node : str
    rel : str
    o_node : str
    """
    s_df = eval("self."+s_node+"s")
    rel_df = eval("self."+s_node+"_"+o_node)
    o_df = eval("self."+o_node+"s")

    # if o_node == 'assay':
    #   # REMOVE WHEN node2 == 2783
    #   rel_df = rel_df.loc[rel_df.node2 != 2783,:]
    
    filtered_rels = rel_df.loc[rel_df.edge == rel,:]

    s_idx_map = dict(zip(s_df.node, s_df.index.tolist()))
    o_idx_map = dict(zip(o_df.node, o_df.index.tolist()))

    s_conv = [s_idx_map[x] for x in filtered_rels.node1.values]  # 'node1' is subject
    o_conv = [o_idx_map[x] for x in filtered_rels.node2.values]  # 'node2' is object

    adj = scipy.sparse.lil_matrix( (max(s_conv)+1, max(o_conv)+1) )  # Add 1 for zero-based indexing

    for i in range(len(s_conv)):
      adj[s_conv[i],o_conv[i]] = 1

    return scipy.sparse.csr_matrix(adj)

  def build_adjacency_matrices(self):
    print("  ...constructing adjacency matrices.")
    metaedges = {
      'chemicalhasactiveassay': ('chemical', 'CHEMICALHASACTIVEASSAY', 'assay'),
      'chemicalhasinactiveassay': ('chemical', 'CHEMICALHASINACTIVEASSAY', 'assay'),
      'chemicalbindsgene': ('chemical', 'CHEMICALBINDSGENE', 'gene'),
      'chemicaldecreasesexpression': ('chemical', 'CHEMICALDECREASESEXPRESSION', 'gene'),
      'chemicalincreasesexpression': ('chemical', 'CHEMICALINCREASESEXPRESSION', 'gene'),
      'geneinteractswithgene': ('gene', 'GENEINTERACTSWITHGENE', 'gene')
    }

    self.adjacency = dict()
    for k, (s,r,o) in metaedges.items():
      self.adjacency[(s,r,o)] = self.make_adjacency(s, r, o)

  def make_heterograph(self):
    print("  ...finalizing heterogeneous graph.")
    graph_data = {
      ('chemical', 'chemicalhasinactiveassay', 'assay'): self.adjacency[('chemical', 'CHEMICALHASINACTIVEASSAY', 'assay')].nonzero(),
      ('assay', 'assayinactiveforchemical', 'chemical'): self.adjacency[('chemical', 'CHEMICALHASINACTIVEASSAY', 'assay')].transpose().nonzero(),
      ('chemical', 'chemicalhasactiveassay', 'assay'): self.adjacency[('chemical', 'CHEMICALHASACTIVEASSAY', 'assay')].nonzero(),
      ('assay', 'assayactiveforchemical', 'chemical'): self.adjacency[('chemical', 'CHEMICALHASACTIVEASSAY', 'assay')].transpose().nonzero(),
      ('chemical', 'chemicalbindsgene', 'gene'): self.adjacency[('chemical', 'CHEMICALBINDSGENE', 'gene')].nonzero(),
      ('gene', 'genebindedbychemical', 'chemical'): self.adjacency[('chemical', 'CHEMICALBINDSGENE', 'gene')].transpose().nonzero(),
      ('chemical', 'chemicaldecreasesexpression', 'gene'): self.adjacency[('chemical', 'CHEMICALDECREASESEXPRESSION', 'gene')].nonzero(),
      ('gene', 'expressiondecreasedbychemical', 'chemical'): self.adjacency[('chemical', 'CHEMICALDECREASESEXPRESSION', 'gene')].transpose().nonzero(),
      ('chemical', 'chemicalincreasesexpression', 'gene'): self.adjacency[('chemical', 'CHEMICALINCREASESEXPRESSION', 'gene')].nonzero(),
      ('gene', 'expressionincreasedbychemical', 'chemical'): self.adjacency[('chemical', 'CHEMICALINCREASESEXPRESSION', 'gene')].transpose().nonzero(),
      ('gene', 'geneinteractswithgene', 'gene'): self.adjacency[('gene', 'GENEINTERACTSWITHGENE', 'gene')].nonzero(),
      ('gene', 'geneinverseinteractswithgene', 'gene'): self.adjacency[('gene', 'GENEINTERACTSWITHGENE', 'gene')].transpose().nonzero(),
    }

    self.G = heterograph(graph_data)

    # Get rid of all links to assays
    # self.G = edge_type_subgraph(self.G, [
    #   ('chemical', 'chemicalbindsgene', 'gene'),
    #   ('gene', 'genebindedbychemical', 'chemical'),
    #   ('chemical', 'chemicaldecreasesexpression', 'gene'),
    #   ('gene', 'expressiondecreasedbychemical', 'chemical'),
    #   ('chemical', 'chemicalincreasesexpression', 'gene'),
    #   ('gene', 'expressionincreasedbychemical', 'chemical'),
    #   ('gene', 'geneinteractswithgene', 'gene'),
    #   ('gene', 'geneinverseinteractswithgene', 'gene'),
    # ])


  def save(self):
    print("Saving data...")
    output_filename = path.join("data", "graph.bin")
    save_graphs(output_filename, self.G)

  def load(self):
    print("Loading data from disk...")
    return load_graphs("../data/graph.bin")[0][0]

  def make_rel_t(self, s_node, rel, o_node):
    """Make a tensor representing links between a subject and an object node
    type for a specified edge type.

    Parameters
    ----------
    s_node : str
    rel : str
    o_node : str
    """
    # Dynamically retrieve dataframes based on subject and object names
    s_df = eval("self."+s_node+"s")
    rel_df = eval("self."+s_node+"_"+o_node)
    o_df = eval("self."+o_node+"s")

    filtered_rels = rel_df.loc[rel_df.edge == rel,:]

    s_idx_map = dict(zip(s_df.node, s_df.index.tolist()))
    o_idx_map = dict(zip(o_df.node, o_df.index.tolist()))

    s_conv = [s_idx_map[x] for x in filtered_rels.node1.values]  # 'node1' is subject
    o_conv = [o_idx_map[x] for x in filtered_rels.node2.values]  # 'node2' is object

    s = torch.tensor(s_conv)
    o = torch.tensor(o_conv)

    return (s, o)

# Add node features
# g.nodes['chemical'].data['maccs'] = maccs_tensor
# g.nodes['gene'].data['dummy'] = torch.ones(g.num_nodes('gene'), 1)
# g.nodes['assay'].data['dummy'] = torch.ones(g.num_nodes('assay'), 1)

# dgl.save_graphs("E:\\data\\comptoxai\\psb\\graph.bin", g)

if __name__=="__main__":
  dset = QSARDataset(name="qsar-gnn")
  ipdb.set_trace()