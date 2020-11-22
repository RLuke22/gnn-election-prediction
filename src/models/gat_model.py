import torch 
torch.manual_seed(22)
import torch.nn as nn 
import torch.nn.functional as F 
from torch_geometric.nn import GATConv

use_gpus = torch.cuda.is_available()
if use_gpus:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class GAT(nn.Module):
    def __init__(self, args):
        super(GAT, self).__init__()
        self.input_dim = args.input_dim 
        self.output_dim = args.output_dim 
        self.heads = args.heads
        self.dropout = args.dropout

        # We use the Transductive settings laid out in the paper: Graph Attention Networks (Petar Velickovic et al., 2017)
        self.conv1 = GATConv(in_channels=self.input_dim, out_channels=self.output_dim, heads=self.heads, dropout=self.dropout)
        self.conv2 = GATConv(self.output_dim * self.heads, 2, heads=1, concat=False, dropout=self.dropout)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

