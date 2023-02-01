import networkx as nx
import dgl
import igraph
import torch as th
from ogb.graphproppred import GraphPropPredDataset


dataset = GraphPropPredDataset(name="ogbg-molhiv")

# dataset[0] returns a tuple <graph, associated label of the graph>
graph, label = dataset[0]
nx_graph = nx.Graph()
nx_graph.add_edges_from(graph["edge_index"].T)

total_edges = int(len(graph["edge_index"][0]) / 2)

totalnodes = graph["num_nodes"]
khop = 3
startnode = int(totalnodes / 2)

nx_subgraph = nx.ego_graph(nx_graph, startnode, radius=khop)

ig_subgraph = igraph.Graph.from_networkx(nx_subgraph)

perm = ig_subgraph.canonical_permutation()

# This line is where the problem is, you used n=num_edges, instead of num_nodes
# ig = igraph.Graph(n=g.num_edges(), edges = listedges, directed = True)


dgl_subgraph = dgl.from_networkx(nx_subgraph)

print(perm)

dgl_subgraph.ndata["canon"] = th.Tensor(perm)

print(dgl_subgraph.ndata["canon"])
