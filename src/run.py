import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys
import argparse 
import time 
import random
random.seed(22)
from numpy.random import seed
seed(22)

import numpy as np
import torch
import torch.nn as nn
torch.manual_seed(22)
from torch import optim 
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchtext import data

from tqdm import tqdm 
import numpy as np
import pickle

from models.bigru_model import BIGRU

from datasets.sentiment140_dataset import Sentiment140Dataset
from datasets.election2020_dataset import Election2020Dataset

# for testing, print full numpy arrays
np.set_printoptions(threshold=sys.maxsize)

use_gpus = torch.cuda.is_available()
if use_gpus:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print('GPU Enabled: {}'.format(use_gpus))

# borrowed from https://github.com/clairett/pytorch-sentiment-classification/
def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1).decode('latin-1')
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs

# borrowed from https://github.com/clairett/pytorch-sentiment-classification
def load_iterators(text_field, label_field, batch_size, fold_n, csv_dir):
    train, val, test = data.TabularDataset.splits(path=csv_dir, train='train{:02d}.csv'.format(fold_n),
                                                  validation='valid{:02d}.csv'.format(fold_n), test='test{:02d}.csv'.format(fold_n), format='csv',
                                                  fields=[('text', text_field), ('label', label_field)])
    text_field.build_vocab(train, val, test)
    label_field.build_vocab(train, val, test)

    train_iter, val_iter, test_iter = data.BucketIterator.splits((train, val, test),
                batch_sizes=(batch_size, batch_size, batch_size), sort_key=lambda x: len(x.text), repeat=False, device=device)
    
    return train_iter, val_iter, test_iter

def make_output_path(args):

    model_dir = '{}'.format(args.model)
    model_dir += '_{}_{}'.format(args.dropout, args.hidden_dim)

    # create path for model weights and results
    out_dir = os.path.join(args.output, model_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    return out_dir

def reduce_lr_on_plateau(optimizer, factor=.1):
    for i, param_group in enumerate(optimizer.param_groups):
        old_lr = float(param_group['lr'])
        new_lr = old_lr * factor
        param_group['lr'] = new_lr
        print('Adjusting learning rate from {:.3e} to {:.3e}'.format(old_lr, new_lr))

# borrowed from https://github.com/clairett/pytorch-sentiment-classification
def get_accuracy(truth, pred):
    assert len(truth) == len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i] == pred[i]:
            right += 1.0
    return right / len(truth)

# borrowed from https://github.com/clairett/pytorch-sentiment-classification
def train_epoch(model, 
                train_iter, 
                loss_function, 
                optimizer, 
                text_field, 
                label_field,
                epoch):
    model.train()
    avg_loss = 0.0
    gts = []
    preds = []
    for batch in tqdm(train_iter):
        text, label = batch.text, batch.label
        
        # decrease class labels by 1
        label.data.sub_(1)
        gts += list(label.data)

        # in case batch size smaller
        model.batch_size = int(label.data.shape[0])
        
        pred = model(text)
        pred_label = pred.data.max(1)[1].cpu().numpy()
        preds += [x for x in pred_label]
        model.zero_grad()
        
        loss = loss_function(pred, label)

        avg_loss += float(loss)
        
        loss.backward()
        optimizer.step()
    
    avg_loss /= len(train_iter)
    acc = get_accuracy(gts, preds)
    return avg_loss, acc

# borrowed from https://github.com/clairett/pytorch-sentiment-classification
def evaluate(model,
             data,
             loss_function):
    model.eval()
    avg_loss = 0.0
    gts = []
    preds = []
    for batch in data:
        text, label = batch.text, batch.label
        # decrease class labels by 1
        label.data.sub_(1)
        gts += list(label.data)
        # in case batch size smaller
        model.batch_size = int(label.data.shape[0])
        
        pred = model(text)
        pred_label = pred.data.max(1)[1].cpu().numpy()
        preds += [x for x in pred_label]
        
        loss = loss_function(pred, label)
        avg_loss += float(loss)
    
    avg_loss /= len(data)
    acc = get_accuracy(gts, preds)
    return avg_loss, acc

def run(args):

    N_SPLITS = 5
    if args.dataset == 'sentiment140':
        data_loader = Sentiment140Dataset(args, N_SPLITS)
        if args.text_cleaning: 
            csv_dir = '../../sentiment140_splits_cleaned'
        else:
            csv_dir = '../../sentiment140_splits'
    
    elif args.dataset == 'election2020':
        data_loader = Election2020Dataset(args, N_SPLITS)
        if args.text_cleaning: 
            csv_dir = '../../election2020_splits_cleaned'
        else:
            csv_dir = '../../election2020_splits'

    # Output path we write best model to
    output_path = make_output_path(args)
    
    # creates csv files which are then passed into the TabularDataset TorchText function call
    data_loader.gen_splits()
    
    for fold_n in range(N_SPLITS):
        best_model_file = os.path.join(output_path, 'fold{:02d}.pth'.format(fold_n))
        
        if args.model == 'bigru':
            text_field = data.Field(lower=True)
            label_field = data.Field(sequential=False)
        
            train_iter, val_iter, test_iter = load_iterators(text_field, label_field, args.batch_size, fold_n, csv_dir)

            # pretrained embeddings -- word2vec
            word_to_idx = text_field.vocab.stoi 
            pretrained_embeddings = np.random.uniform(-0.25, 0.25, (len(text_field.vocab), 300))
            pretrained_embeddings[0] = 0 
            # load word2vec vectors
            word2vec = load_bin_vec(args.word2vec_path, word_to_idx)
            
            # populate the pretrained embeddings
            for word, vector in word2vec.items():
                pretrained_embeddings[word_to_idx[word] - 1] = vector
            
            model = BIGRU(args, vocab_size=len(text_field.vocab)).cuda()
            # populate the weights of the embedding layer
            model.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            loss_function = nn.NLLLoss()

            best_model = model 
            best_val_acc = 0
            early_stop_idx = 0
            reduce_lr_idx = 0
            print("Training Fold {}...".format(fold_n))
            for epoch in range(args.epochs):
                start_time = time.time()
                loss, acc = train_epoch(
                    model, 
                    train_iter, 
                    loss_function, 
                    optimizer, 
                    text_field, 
                    label_field,
                    epoch
                )
                # logging
                end_time = time.time()
                print("Epoch: {}".format(epoch))
                print("Time Elapsed: {:.2f}s".format(end_time - start_time))
                print("Train - Loss: {:.3f}, Accuracy: {:.2f}".format(loss, acc*100))

                with torch.no_grad():
                    val_loss, val_acc = evaluate(model, val_iter, loss_function)
                    print("Validation - Loss: {:.3f}, Accuracy: {:.2f}".format(val_loss, val_acc*100))
                    print("Learning rate: {:.3e}".format(optimizer.param_groups[0]['lr']))
                    
                    if val_acc > best_val_acc:
                        print("Validation accuracy improved from {:.5f} to {:.5f}. Saving model to {}".format(best_val_acc, val_acc, best_model_file))
                        
                        best_val_acc = val_acc
                        best_model = model

                        torch.save({
                            "epoch": epoch+1,
                            "best_val_acc": best_val_acc,
                            "model": best_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        }, best_model_file)

                        early_stop_idx = 0
                        reduce_lr_idx = 0
                    else:
                        print("Validation acc did not improve from {:.5f}".format(best_val_acc))
                        early_stop_idx += 1
                        reduce_lr_idx += 1

                    # Learning Rate Decay
                    if reduce_lr_idx > args.reduce_lr - 1:
                        reduce_lr_factor = 0.1
                        reduce_lr_on_plateau(optimizer=optimizer, factor=reduce_lr_factor)
                        reduce_lr_idx = 0

                    # Early Stopping
                    if early_stop_idx > args.early_stopping - 1:
                        print('\nEarly stopping at epoch {}'.format(epoch+1))
                        break
                    
                    print()

            print("Evaluating Fold {}...".format(fold_n))
            test_loss, test_acc = evaluate(best_model, test_iter, loss_function)
            print("Test Accuracy: {:.5f}".format(test_acc))
            print()



def read_args(args):
    parser = argparse.ArgumentParser(description=__doc__)

    # model selection parameters
    parser.add_argument('--model', dest='model', type=str, default='bigru', help='bigru')
    parser.add_argument('--dataset', dest='dataset', type=str, default='election2020', help='election2020,sentiment140')
    parser.add_argument('--seed', dest='seed', type=int, default=22)
    parser.add_argument('--output-dir', dest='output', type=str, default='results')
    parser.add_argument('--word2vec-path', dest='word2vec_path', type=str, default='../../GoogleNews-vectors-negative300.bin')

    # text parameters
    parser.add_argument('--text-cleaning', dest='text_cleaning', action='store_true', help='clean the data?')

    # model training parameters
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=32)
    parser.add_argument('--epochs', dest='epochs', type=int, default=2)
    parser.add_argument('--early-stopping', dest='early_stopping', type=int, default=3)
    parser.add_argument('--lr', dest='lr', type=float, default=2e-5)
    parser.add_argument('--reduce-lr', dest='reduce_lr', type=int, default=2)
    parser.add_argument('--num-workers', dest='num_workers', type=int, default=6)
    
    # model parameters
    parser.add_argument('--embedding-dim', dest='embedding_dim', type=int, default=300)
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=96)
    parser.add_argument('--dropout', dest='dropout', type=float, default=0)

    return parser.parse_args(args)

if __name__ == '__main__':
    args = read_args(sys.argv[1:])

    print("Experiment configurations")
    print("_____________________")
    print("Model: ", args.model)
    print("Early Stopping: ", args.early_stopping)
    print("Reduce Learning Rate: ", args.reduce_lr)
    print("Learning Rate: ", args.lr) 
    print("Batch size: ", args.batch_size)
    print()

    run(args)