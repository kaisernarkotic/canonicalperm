import networkx as nx
from ogb.graphproppred import GraphPropPredDataset

dataset = GraphPropPredDataset(name = "ogbg-molhiv")

graph, label = dataset[0]
nx_G = nx.Graph()
nx_G.add_edges_from(graph["edge_index"].T)

total_edges = int(graph.get("edge_index").size / 2)

totalnodes = graph.get("num_nodes")
khop = 3

import dgl
import numpy as np
import igraph
import torch as th

subgraph_list = []
sg_per_g = [graph.get("num_nodes")] #and if there is a graph2, label = dataset[1], then this will also be in the subgraphs per graph list
# which will be used later down the road in the dgl.readout_nodes 

for i in range(0, 10):
  startnode = i

  nx_subg = nx.ego_graph(nx_G, startnode, radius = khop)
  
  ig = igraph.Graph.from_networkx(nx_subg)

  cperm = ig.canonical_permutation(sh = "f", color = None)

  dgl_subg = dgl.from_networkx(nx_subg)

  dgl_subg.ndata['canon'] = th.Tensor(cperm)

  subgraph_list.append(dgl_subg)


batched_graphs = dgl.batch(subgraph_list)

"""
for i, graph in enumerate(modified_subgraphs):
  print(f"Subgraph {i}: {graph.number_of_nodes()} nodes")
"""
import dgl
import dgl.data
import torch.nn as nn
import torch.nn.functional as F
import os

class GCN(nn.Module):
  def __init__(self, in_feats, h_feats, num_classes):
    super(GCN, self).__init__()
    self.conv1 = dgl.nn.GraphConv(in_feats, h_feats)
    self.conv2 = dgl.nn.GraphConv(h_feats, num_classes)
  def forward(self, g, in_feat):
    h = self.conv1(g, in_feat)
    h = F.relu(h)
    h = self.conv2(g, h)
    h = F.relu(h)
    with g.local_scope():
      g.ndata["h"] = h
      hg = dgl.readout_nodes(g, "h", op='sum')
      g1 = th.split(hg, split_size_or_secion=sg_per_g[0], dim=0)
      sum = th.sum(g1)
      return self.classify(sum) 
  def classify(self, x):
    with torch.no_grad():
      logits = self.forward(x)
      preds = torch.argmax(logits, dim=1)
      return preds
