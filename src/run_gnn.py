import torch
import os
import time
import sys
import argparse
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv
from datasets.create_gnn_dataset import GNNDataset
from torch_geometric.data import Data
from models.gat_model import GAT
import numpy as np

use_gpus = torch.cuda.is_available()
if use_gpus:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def reduce_lr_on_plateau(optimizer, factor=.1):
    for i, param_group in enumerate(optimizer.param_groups):
        old_lr = float(param_group['lr'])
        new_lr = old_lr * factor
        param_group['lr'] = new_lr
        print('Adjusting learning rate from {:.3e} to {:.3e}'.format(old_lr, new_lr))

def make_output_path(args):
    model_dir = '{}'.format(args.model)
    model_dir += '_{}_{}_{}'.format(args.heads, args.output_dim, args.dropout)

    # create path for model weights and results
    out_dir = os.path.join(args.gru_output_dir, model_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    return out_dir


def train(data, model, optimizer):
    model.train()
    optimizer.zero_grad()

    logits = model(data.x, data.edge_index)[data.train_mask]
    pred = logits.max(1)[1]
    train_acc = pred.eq(data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()

    F.nll_loss(logits, data.y[data.train_mask]).backward()
    optimizer.step()

    return train_acc

def inference(data, model, test_d_mask, test_r_mask):
    model.eval()
    logits, accs = model(data.x, data.edge_index), []
    i = 0
    for _, mask in data('val_mask', 'test_mask'):
        logits_masked = logits[mask]
        pred = logits_masked.max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)

        if i == 1:
            pred_d = logits[test_d_mask].max(1)[1]
            pred_r = logits[test_r_mask].max(1)[1]
            acc_d = pred_d.eq(data.y[test_d_mask]).sum().item() / test_d_mask.sum().item()
            acc_r = pred_r.eq(data.y[test_r_mask]).sum().item() / test_r_mask.sum().item()
            accs.append(acc_d)
            accs.append(acc_r)
        i += 1

    return accs

def run(args):

    N_SPLITS = 5
    output_path = make_output_path(args)

    gnn_dataset = GNNDataset(args) 
    five_fold_test_acc = []
    five_fold_test_acc_d = []
    five_fold_test_acc_r = []
    for fold_n in range(N_SPLITS):
        data, test_d_mask, test_r_mask = gnn_dataset.create_gnn_dataset(fold_n)
        model, data = GAT(args).to(device), data.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_val_acc = 0
        best_test_acc
        best_val_acc_d = 0
        best_val_acc_r = 0
        early_stop_idx = 0
        reduce_lr_idx = 0
        for epoch in range(1, args.epochs+1):
            start_time = time.time()
            train_acc = train(data, model, optimizer)
            val_acc, test_acc, test_acc_d, test_acc_r,  = inference(data, model, test_d_mask, test_r_mask)
            end_time = time.time()
            print('Epoch: {:03d}, Time: {:.2f}s, Train: {:.4f}, Val: {:.4f}, Val_D: {:.4f}, Val_R: {:.4f}, Test: {:.4f}'.format(epoch, end_time-start_time, train_acc, val_acc, val_acc_d, val_acc_r, test_acc))

            if val_acc > best_val_acc:
                print("Validation accuracy improved from {:.5f} to {:.5f}".format(best_val_acc, val_acc))
                best_val_acc = val_acc
                best_test_acc = test_acc
                best_test_acc_d = test_acc_d 
                best_test_acc_r = test_acc_r
                early_stop_idx = 0
                reduce_lr_idx = 0
            else:
                print("Validation accuracy did not improve from {:.5f}".format(best_val_acc))
                early_stop_idx += 1
                reduce_lr_idx += 1
            print()

            # Learning Rate Decay
            if reduce_lr_idx > args.reduce_lr - 1:
                reduce_lr_factor = 0.1
                reduce_lr_on_plateau(optimizer=optimizer, factor=reduce_lr_factor)
                reduce_lr_idx = 0

            # Early Stopping
            if early_stop_idx > args.early_stopping - 1:
                print('\nEarly stopping at epoch {}'.format(epoch+1))
                five_fold_test_acc.append(best_test_acc)
                five_fold_test_acc_d.append(best_test_acc_d)
                five_fold_test_acc_r.append(best_test_acc_r)
                break
        print()
    print("Final Results")
    print("------------------------------------")
    print("Test Accuracy: ", np.mean(np.array(five_fold_test_acc)))
    print("Test Accuracy-D: ", np.mean(np.array(five_fold_test_acc_d)))
    print("Test Accuracy-R: ", np.mean(np.array(five_fold_test_acc_r)))
    print("------------------------------------")


def read_args(args):
    parser = argparse.ArgumentParser(description=__doc__)

    # model selection parameters
    parser.add_argument('--model', dest='model', type=str, default='GAT', help='GAT')
    parser.add_argument('--seed', dest='seed', type=int, default=22)
    parser.add_argument('--gru-output-dir', dest='gru_output_dir', type=str, default='results/bigru_96')

    # model training parameters
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=128)
    parser.add_argument('--epochs', dest='epochs', type=int, default=50)
    parser.add_argument('--lr', dest='lr', type=float, default=0.005)
    parser.add_argument('--weight-decay', dest='weight_decay', type=float, default=5e-4)
    parser.add_argument('--early-stopping', dest='early_stopping', type=int, default=5)
    parser.add_argument('--reduce-lr', dest='reduce_lr', type=int, default=1)
    
    # model parameters
    parser.add_argument('--input-dim', dest='input_dim', type=int, default=194)
    parser.add_argument('--heads', dest='heads', type=int, default=8)
    parser.add_argument('--output-dim', dest='output_dim', type=int, default=8)
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.6)

    return parser.parse_args(args)

if __name__ == '__main__':
    args = read_args(sys.argv[1:])

    print("Experiment configurations")
    print("_____________________")
    print("Epochs", args.epochs)
    print("Learning Rate: ", args.lr) 
    print("Batch size: ", args.batch_size)
    print("Weight decay: ", args.weight_decay)
    print("Input dim: ", args.input_dim)
    print("Output dim: ", args.output_dim)
    print("Heads: ", args.heads)
    print("Dropout: ", args.dropout)
    print("Reduce LR: ", args.reduce_lr)
    print("Early Stopping: ", args.early_stopping)
    print()

    run(args)