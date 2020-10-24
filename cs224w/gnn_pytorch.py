import torch 
import torch.nn as nn 
import torch.nn.functional as F 

# Used for graph neural networks
import torch_geometric.nn as pyg_nn 
import torch_geometric.utils as pyg_utils 

import time 
from datetime import datetime 

# Used for graph visualization
import networkx as nx 
import numpy as np 
import torch 
import torch.optim as optim 

from torch_geometric.datasets import TUDataset 
from torch_geometric.datasets import Planetoid 
from torch_geometric.data import DataLoader 

import torch_geometric.transforms as T 

# More for visualizations
# from tensorboardX import SummaryWriter 
# embed into 2 dimensions
# from sklearn.manifold import TSNE 
# import matplotlib.pyplot as plt

# Simple GNN model
class GNNStack(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, task='node'):
        super(GNNStack, self).__init__()
        self.task = task 
        # Put all convolutions into a module
        self.convs = nn.ModuleList() 

        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        for l in range(2):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))

        # post message passing
        # Sequential always executes in the order prescibed
        self.post_mp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.25),
                nn.Linear(hidden_dim, output_dim))

        if not (self.task == 'node' or self.task == 'graph'):
            raise RunTimeError('Unknown task.')

        self.dropout = 0.25 
        # This indicates the depth of the neural network
        self.num_layers = 3

    def build_conv_model(self, input_dim, hidden_dim):
        # if it is node classifications, use vanilla graph convolution
        if self.task == 'node':
            return pyg_nn.GCNConv(input_dim, hidden_dim)
        # If it is a graph classification 
        else:
            return pyg_nn.GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                    nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))

    # Specific to graph neural networks
    def forward(self, data):
        # x is the feature matrix (n_nodes x embed_dim)
        # edge_index is the adjacency list of the graph
        # batch: for every node indexing this array, you record which graph it belongs to
        # Oftentimes just an all 1's array
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # All 1's when no features for node
        if data.num_node_features == 0:
            x = torch.ones(data.num_nodes, 1)

        # Layers that execute the convolution
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            emb = x 
            x = F.relu(x) 
            x = F.dropout(x, p=self.dropout, training=self.training)

        # need to pool nodes when doing graph classification
        if self.task == 'graph':
            x = pyg_nn.global_mean_pool(x, batch)

        x = self.post_mp(x)

        return emb, F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


def train(dataset, task, writer):
    # Should extent with validation set (with cross validation)
    # 0.8 train / 0.1 validation / 0.1 test
    if task == 'graph':
        data_size = len(dataset) 
        loader = DataLoader(dataset[:int(data_size * 0.8)], batch_size=64, shuffle=True)
        test_loader = DataLoader(dataset[int(data_size * 0.8):], batch_size=64, shuffle=True)
    else:
        test_loader = loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # build model 
    model = GNNStack(max(dataset.num_node_features, 1), 32, dataset.num_classes, task=task)
    opt = optim.Adam(model.parameters(), lr=0.01)

    # train
    for epoch in range(200):
        total_loss = 0 
        model.train() 
        for batch in loader:
            opt.zero_grad() 
            embedding, pred = model(batch) 
            label = batch.y 

            # A mask which indicates which nodes are for training
            if task == 'node':
                pred = pred[batch.train_mask]
                label = label[batch.train_mask] 

            loss = model.loss(pred, label)
            loss.backward()
            opt.step()

            # batch.num_graphs will be 1 for node classification
            total_loss += loss.item() * batch.num_graphs

        total_loss /= len(loader.dataset) 
        # writer.add_scalar("loss", total_loss, epoch)

        if epoch % 10 == 0:
            test_acc = test(test_loader, model) 
            print("Epoch {}. Loss: {:.4f}. Test accuracy: {:.4f}".format(epoch, total_loss, test_acc))

            # writer.add_scalar("test accuracy", test_acc, epoch)

    return model

def test(loader, model, is_validation=False):
    model.eval()

    correct = 0 
    for data in loader: 
        with torch.no_grad():
            emb, pred = model(data) 
            pred = pred.argmax(dim=1) 
            label = data.y  

        if model.task == 'node':
            mask = data.val_mask if is_validation else data.test_mask 
            # node classification: only evaluate on nodes in the test set 
            pred = pred[mask] 
            label = data.y[mask] 

        correct += pred.eq(label).sum().item()

    if model.task == 'graph':
        total = len(loader.dataset) 
    else:
        total = 0 
        for data in loader.dataset:
            total += torch.sum(data.test_mask).item() 
        
    return correct / total

# For an example, we will perform node classification on the Citesser citation network 
dataset = Planetoid(root='./', name='citeseer')
task = 'node'

model = train(dataset, task, None)