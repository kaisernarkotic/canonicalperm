import dgl
import numpy as np
import igraph
import torch as th
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
import os
from ogb.graphproppred import GraphPropPredDataset
from torch_geometric.loader import DataLoader

dataset = GraphPropPredDataset(name = "ogbg-molhiv")

graph, label = dataset[0]

train_idx = dataset.get_idx_split()['train']
train_data = [dataset[i][0] for i in train_idx]

valid_idx = dataset.get_idx_split()['valid']
valid_data = [dataset[i][0] for i in valid_idx]

test_idx = dataset.get_idx_split()['test']
test_data = [dataset[i][0] for i in test_idx]

train_loader = DataLoader(train_data, 1, shuffle=True)
valid_loader = DataLoader(valid_data, 1, shuffle=False)
test_loader = DataLoader(test_data, 1, shuffle=False)

nx_G = nx.Graph()
nx_G.add_edges_from(graph["edge_index"].T)

total_edges = int(graph.get("edge_index").size / 2)

totalnodes = graph.get("num_nodes")
khop = 3


subgraph_list = []
sg_per_g = [graph.get("num_nodes")] #and if there is a graph2, label = dataset[1], then this will also be in the subgraphs per graph list
# which will be used later down the road in the dgl.readout_nodes

for i in range(0, graph.get("num_nodes")):
  startnode = i

  nx_subg = nx.ego_graph(nx_G, startnode, radius = khop)

  #ig = igraph.Graph.from_networkx(nx_subg)

  #cperm = ig.canonical_permutation(sh = "f", color = None)

  dgl_subg = dgl.from_networkx(nx_subg)

  one_feat = th.ones(dgl_subg.number_of_nodes())

  dgl_subg.ndata['feat'] = one_feat

  #dgl_subg.ndata['canon'] = th.Tensor(cperm)

  subgraph_list.append(dgl_subg)


batched_graphs = dgl.batch(subgraph_list)

#print(batched_graphs.ndata)

class GCN(nn.Module):
  def __init__(self, in_feats, h_feats, num_classes):
    super(GCN, self).__init__()
    self.conv1 = dgl.nn.GraphConv(in_feats, h_feats)
    self.conv2 = dgl.nn.GraphConv(h_feats, num_classes)
  def forward(self, g, in_feat):#, split_info):
    h = self.conv1(g, in_feat)
    h = F.relu(h)
    h = self.conv2(g, h)
    h = F.relu(h)
    g.ndata["h"] = h
    return dgl.mean_nodes(g, "h")
    """with g.local_scope():
      g.ndata['canon'] = h
      hg = dgl.readout_nodes(g, 'canon', op='sum')
      g1 = th.split(hg, split_size_or_section=split_info[0], dim=0)
      sum = 0
      for x in g1:
        sum+=th.sum(x)
      return sum"""

class Classify(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(16, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)

in_feats = 1
out_feats = 16
model = GCN(in_feats, 16, out_feats)
# Separate neural network
mlp = Classify()
# Loss function
loss_fn = nn.CrossEntropyLoss()
# Define the optimizer
optimizer = th.optim.Adam(model.parameters(), lr=0.01)

# Train the model
for epoch in range(20):
    for labels in label:
        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model.forward(batched_graphs, "feat") #, 'canon', sg_per_g)
        passed_through = mlp.forward(outputs)
        # Compute loss
        loss = loss_fn(passed_through, labels)
        # Backward pass
        loss.backward()
        # Update parameters
        optimizer.step()

num_correct = 0
num_tests = 0
for batched_graphs, labels in test_loader:
    pred = model(batched_graphs, batched_graphs.ndata["feat"].float())
    num_correct += (pred.argmax(1) == labels).sum().item()
    num_tests += len(labels)

print("Test accuracy:", num_correct / num_tests)