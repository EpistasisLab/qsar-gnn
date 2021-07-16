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
import dgl
import pandas as pd
import numpy as np
import ipdb

chemicals = pd.read_csv("E:\\data\\comptoxai\\psb\\chemicals.csv")
genes = pd.read_csv("E:\\data\\comptoxai\\psb\\genes.csv")
assays = pd.read_csv("E:\\data\\comptoxai\\psb\\assays.csv")

maccs_ndarray = np.empty(shape=(len(chemicals), len(chemicals.maccs[0])))
for i, m in enumerate(chemicals.maccs):
  maccs_ndarray[i,:] = [int(mm) for mm in m]
maccs_tensor = torch.tensor(maccs_ndarray, dtype=torch.bool)

chemical_assay = pd.read_csv("E:\\data\\comptoxai\\psb\\chemical-assay.csv")
chemical_gene = pd.read_csv("E:\\data\\comptoxai\\psb\\chemical-gene.csv")
gene_gene = pd.read_csv("E:\\data\\comptoxai\\psb\\gene-gene.csv")

def make_rel_t(n1, n2, data, rel):
  filtered = data.loc[data.edge == rel,:]
  
  # We want to map node IDs to the indices of the nodes in the dataframe
  n1_idx_map = dict(zip(n1.node, n1.index.tolist()))
  n2_idx_map = dict(zip(n2.node, n2.index.tolist()))

  try:
    n1_conv = [n1_idx_map[x] for x in filtered.node1.values]
    n2_conv = [n2_idx_map[x] for x in filtered.node2.values]
  except KeyError:
    ipdb.set_trace()
    print()
  
  n1 = torch.tensor(n1_conv)
  n2 = torch.tensor(n2_conv)

  return (n1, n2)

graph_data = {
  ('chemical', 'chemicalhasinactiveassay', 'assay'): make_rel_t(chemicals, assays, chemical_assay, 'CHEMICALHASINACTIVEASSAY'),
  ('chemical', 'chemicalhasactiveassay', 'assay'): make_rel_t(chemicals, assays, chemical_assay, 'CHEMICALHASACTIVEASSAY'),
  ('chemical', 'chemicalbindsgene', 'gene'): make_rel_t(chemicals, genes, chemical_gene, 'CHEMICALBINDSGENE'),
  ('chemical', 'chemicaldecreasesexpression', 'gene'): make_rel_t(chemicals, genes, chemical_gene, 'CHEMICALDECREASESEXPRESSION'),
  ('chemical', 'chemicalincreasesexpression', 'gene'): make_rel_t(chemicals, genes, chemical_gene, 'CHEMICALINCREASESEXPRESSION'),
  ('gene', 'geneinteractswithgene', 'gene'): make_rel_t(genes, genes, gene_gene, 'GENEINTERACTSWITHGENE'),
}

g = dgl.heterograph(graph_data)

# Add node features
g.nodes['chemical'].data['maccs'] = maccs_tensor
g.nodes['gene'].data['dummy'] = torch.ones(g.num_nodes('gene'), 1)
g.nodes['assay'].data['dummy'] = torch.ones(g.num_nodes('assay'), 1)

dgl.save_graphs("E:\\data\\comptoxai\\psb\\graph.bin", g)