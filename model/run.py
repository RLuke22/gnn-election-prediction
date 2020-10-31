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
torch.manual_seed(22)
from torch import optim 
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tqdm import tqdm 
import numpy as np
import pickle
import pandas as pd

from preprocess import DataPreprocessor

# for testing, print full numpy arrays
np.set_printoptions(threshold=sys.maxsize)

use_gpus = torch.cuda.is_available()
if use_gpus:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print('GPU Enabled: {}'.format(use_gpus))

def train_model(args):

    data_preprocessor = DataPreprocessor(args)

    if args.dataset == 'sentiment140':
        text, labels = data_preprocessor.preprocess_sentiment140()
    elif args.dataset == 'election2020':
        text, labels = data_preprocessor.preprocess_election2020()

    if args.text_cleaning:
        text = data_preprocessor.clean(text)

    

    





def read_args(args):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--model', dest='model', type=str, help='lstm, bert, bert-sentence', default='rowe')
    parser.add_argument('--dataset', dest='dataset', type=str, default='election2020', help='election2020,sentiment140')
    parser.add_argument('--seed', dest='seed', type=int, default=22)
    parser.add_argument('--output-dir', dest='output', type=str, default='results')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=32)
    parser.add_argument('--epochs', dest='epochs', type=int, default=200)
    parser.add_argument('--early-stopping', dest='early_stopping', type=int, default=20)
    parser.add_argument('--lr', dest='lr', type=float, default=0.001)
    parser.add_argument('--reduce-lr', dest='reduce_lr', type=int, default=5)
    parser.add_argument('--num-workers', dest='num_workers', type=int, default=6)
    parser.add_argument('--text-cleaning', dest='text_cleaning', action='store_true', help='clean the data?')

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

    train_model(args)
    test_model(args)