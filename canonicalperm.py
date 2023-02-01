import networkx as nx
from ogb.graphproppred import GraphPropPredDataset

#Subgraphing Section
def subgraph(startnode, khop):
    subgraph = nx.Graph()
    shortestpaths = nx.shortest_path_length(G, source=startnode, method='dijkstra')
    for j in list(shortestpaths.keys()):
        if shortestpaths[j] <= khop:
            subgraph.add_edge(startnode, shortestpaths[j], weight=1)
            #figure out how to get iterated node in the shortestpaths list
    return subgraph

G = nx.Graph()

dataset = GraphPropPredDataset(name = "ogbg-molhiv")

graph = dataset[0] 

total_edges = int(graph[0].get("edge_index").size / 2)


for i in range(0, total_edges):  
  node1 = graph[0].get("edge_index")[0][i]
  node2 = graph[0].get("edge_index")[1][i]
  G.add_edge(node1, node2, weight = 1)

totalnodes = graph[0].get("num_nodes")
khop = 3

#for i in range(0, totalnodes):
#subg = subgraph(i, khop)
startnode = int(totalnodes/2)
subg = subgraph(startnode, khop)
#print(subg)

#CANNONICAL PERMUTATION SECTION
import dgl
import numpy as np
import igraph
import torch as th

g = dgl.from_networkx(subg)

#print(g)

g_adj_matrix = g.adj_sparse('coo')

fnode = list(g_adj_matrix[0].numpy())
snode = list(g_adj_matrix[1].numpy())

#print(fnode)
#print(snode)

ig_edgenum = len(g_adj_matrix[0])


listedges = []

for x in range(ig_edgenum):
    listedges.append([fnode[x],snode[x]])
#print(listedges)

ig = igraph.Graph(n=g.num_edges(), edges = listedges, directed = True)


cperm = ig.canonical_permutation(sh = "f", color = None) #should return the number nodes not edges

print(cperm)

listedges = ig.get_edgelist()

g = dgl.graph(listedges)

print(g)

g.ndata['canon'] = th.ones(g.num_nodes, cperm)

print(g)