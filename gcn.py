import dgl
import dgl.data
import torch
import torch.nn as nn
import os

dataset = dgl.data.GINDataset("PROTEINS", self_loop=True)

from torch.utils.data.sampler import SubsetRandomSampler

from dgl.dataloading import GraphDataLoader

num_examples = len(dataset)
num_train = int(num_examples * 0.8)

train_sampler = SubsetRandomSampler(torch.arange(num_train))
test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

train_dataloader = GraphDataLoader(
    dataset, sampler=train_sampler, batch_size=5, drop_last=False
)
test_dataloader = GraphDataLoader(
    dataset, sampler=test_sampler, batch_size=5, drop_last=False
)

class GCN(nn.Module):
  def __init__(self, in_feats, h_feats, num_classes):
    super(GCN, self).__init__()
    self.conv1 = dgl.nn.GraphConv(in_feats, hidden_size)
    self.conv2 = dgl.nn.GraphConv(h_feats, num_classes)
  def forward(self, g, in_feat):
    h = self.conv1(h, in_feat)
    h = F.relu(h)
    h = self.conv2(g, h)
    g.ndata["h"] = h
    return dgl.mean_nodes(g, "h")

model = GCN(dataset.dim_nfeats, 16, dataset.gclasses)

loss_fn = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
for epoch in range(num_epochs):
    for batched_graph, labels in dataloader:
        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(batched_graph)
        # Compute loss
        loss = loss_fn(outputs, labels)
        # Backward pass
        loss.backward()
        # Update parameters
        optimizer.step()

num_correct = 0
num_tests = 0
for batched_graph, labels in test_dataloader:
    pred = model(batched_graph, batched_graph.ndata["attr"].float())
    num_correct += (pred.argmax(1) == labels).sum().item()
    num_tests += len(labels)

print("Test accuracy:", num_correct / num_tests)