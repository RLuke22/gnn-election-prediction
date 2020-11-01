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
import pandas as pd

from models.bilstm_model import BILSTM

from datasets.sentiment140_dataset import Sentiment140Dataset
# from datasets.election2020_dataset import Election2020Dataset

# for testing, print full numpy arrays
np.set_printoptions(threshold=sys.maxsize)

use_gpus = torch.cuda.is_available()
if use_gpus:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print('GPU Enabled: {}'.format(use_gpus))

# borrowed from https://github.com/clairett/pytorch-sentiment-classification/blob/master/train_batch.py
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

def load_iterators(text_field, label_field, batch_size, fold_n, csv_dir):
    train, val, test = data.TabularDataset.splits(path=csv_dir, train='train{:02d}.csv'.format(fold_n),
                                                  validation='valid{:02d}.csv'.format(fold_n), test='test{:02d}.csv'.format(fold_n), format='csv',
                                                  fields=[('text', text_field), ('label', label_field)])
    text_field.build_vocab(train, val, test)
    label_field.build_vocab(train, val, test)

    train_iter, val_iter, test_iter = data.BucketIterator.splits((train, val, test),
                batch_sizes=(batch_size, len(val), len(test)), sort_key=lambda x: len(x.text), repeat=False, device=device)
    
    return train_iter, val_iter, test_iter

def make_output_path(args):

    model_dir = '{}'.format(args.model)
    model_dir += '_{}_{}'.format(args.dropout, args.hidden_dim)

    # create path for model weights and results
    out_dir = os.path.join(args.output, model_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    return out_dir

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

    # Output path we write best model to
    output_path = make_output_path(args)
    
    # creates csv files which are then passed into the TabularDataset TorchText function call
    data_loader.gen_splits()
    
    for fold_n in range(N_SPLITS):
        if args.model == 'lstm':
            text_field = data.Field(lower=True)
            label_field = data.Field(sequential=False)
        
            train_iter, dev_iter, test_iter = load_iterators(text_field, label_field, args.batch_size, fold_n, csv_dir)

            # pretrained embeddings -- word2vec
            word_to_idx = text_field.vocab.stoi 
            pretrained_embeddings = np.random.uniform(-0.25, 0.25, (len(text_field.vocab), 300))
            pretrained_embeddings[0] = 0 
            # load word2vec vectors
            word2vec = load_bin_vec(args.word2vec_path, word_to_idx)
            
            # populate the pretrained embeddings
            for word, vector in word2vec.items():
                pretrained_embeddings[word_to_idx[word] - 1] = vector
            
            model = BILSTM(args, vocab_size=len(text_field.vocab)).cuda()
            # populate the weights of the embedding layer
            model.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

            best_model = model 
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            loss_function = nn.NLLLoss()

            print("Training Fold {}...".format(fold_n))
            for epoch in range(args.epochs):
                print("Made it here.")
                exit(0)





def read_args(args):
    parser = argparse.ArgumentParser(description=__doc__)

    # model selection parameters
    parser.add_argument('--model', dest='model', type=str, help='lstm, roberta-sentence', default='lstm')
    parser.add_argument('--dataset', dest='dataset', type=str, default='sentiment140', help='election2020,sentiment140')
    parser.add_argument('--seed', dest='seed', type=int, default=22)
    parser.add_argument('--output-dir', dest='output', type=str, default='results')
    parser.add_argument('--word2vec-path', dest='word2vec_path', type=str, default='../../GoogleNews-vectors-negative300.bin')

    # text parameters
    parser.add_argument('--text-cleaning', dest='text_cleaning', action='store_true', help='clean the data?')

    # model training parameters
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=5)
    parser.add_argument('--epochs', dest='epochs', type=int, default=100)
    parser.add_argument('--early-stopping', dest='early_stopping', type=int, default=10)
    parser.add_argument('--lr', dest='lr', type=float, default=0.001)
    parser.add_argument('--reduce-lr', dest='reduce_lr', type=int, default=5)
    parser.add_argument('--num-workers', dest='num_workers', type=int, default=6)
    
    # model parameters
    parser.add_argument('--embedding-dim', dest='embedding_dim', type=int, default=300)
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=150)
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.5)

    
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