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
import pandas as pd
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
from sklearn.metrics import confusion_matrix

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
def load_iterators(text_field, label_field, follows_d_field, follows_r_field, tweet_index_field, batch_size, fold_n, csv_dir, full_training, save_embeddings, full_inference):

    if full_training:
        train_csv_name = 'full_data.csv'
    else:
        train_csv_name = 'train{:02d}.csv'.format(fold_n)

    if full_inference:
        test_csv_name = 'full_inference_data.csv'
    else:
        test_csv_name = 'test{:02d}.csv'.format(fold_n)

    # # We need all embeddings for GNN. Thus, we still perform 5-fold CV
    # # and use the training data at each fold to train the model. Then
    # # perform inference on all the data (Note full_data.csv is simply a csv containing all data train/val/test)
    # if save_embeddings:
    #     test_csv_name = 'full_data.csv'
    # else:
    #     test_csv_name = 'test{:02d}.csv'.format(fold_n)

    train, val, test = data.TabularDataset.splits(path=csv_dir, train=train_csv_name,
                                                  validation='valid{:02d}.csv'.format(fold_n), test=test_csv_name, format='csv',
                                                  fields=[('text', text_field), ('label', label_field), ('follows_d', follows_d_field), ('follows_r', follows_r_field), ('tweet_index', tweet_index_field)])
    
    text_field.build_vocab(train, val, test)
    label_field.build_vocab(train, val, test)
    follows_d_field.build_vocab(train, val, test)
    follows_r_field.build_vocab(train, val, test)

    train_iter, val_iter, test_iter = data.BucketIterator.splits((train, val, test),
                batch_sizes=(batch_size, batch_size, batch_size), sort_key=lambda x: len(x.text), repeat=False, device=device)
    
    return train_iter, val_iter, test_iter

def make_output_path(args):

    model_dir = '{}'.format(args.model)
    if args.full_training:
        model_dir += '_full'
    if args.reweight_loss:
        model_dir += '_balanced'

    model_dir += '_{}'.format(args.hidden_dim)

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
        text, label, follows_d, follows_r = batch.text, batch.label, batch.follows_d, batch.follows_r
        
        # decrease class labels by 1
        label.data.sub_(1)
        gts += list(label.data)

        # in case batch size smaller
        model.batch_size = int(label.data.shape[0])
        
        pred, _ = model(text, follows_d, follows_r)
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
def val_evaluate(model,
             data,
             loss_function):
    model.eval()
    avg_loss = 0.0
    gts = []
    preds = []
    for batch in data:
        text, label, follows_d, follows_r = batch.text, batch.label, batch.follows_d, batch.follows_r
        # decrease class labels by 1
        label.data.sub_(1)
        gts += list(label.data)
        # in case batch size smaller
        model.batch_size = int(label.data.shape[0])
        
        pred, _ = model(text, follows_d, follows_r)
        pred_label = pred.data.max(1)[1].cpu().numpy()
        preds += [x for x in pred_label]
        
        loss = loss_function(pred, label)
        avg_loss += float(loss)
    
    avg_loss /= len(data)
    acc = get_accuracy(gts, preds)
    
    return avg_loss, acc

# borrowed from https://github.com/clairett/pytorch-sentiment-classification
def test_evaluate(model,
                data,
                loss_function,
                save_embeddings,
                hidden_dim):
    
    model.eval()
    avg_loss = 0.0
    gts = []
    tweet_indices = []
    if save_embeddings:
        # [n_tweets x 2*hidden_dim] (where hidden_dim is hidden dimension in GRU)
        # TODO: remove hardcoded integer
        sentence_embeddings = torch.zeros(122443, hidden_dim * 2)
    
    softmax_scores = torch.zeros(655027, 2)
    preds = []
    count = 0
    count_softmax = 0
    for batch in tqdm(data):
        text, label, follows_d, follows_r, tweet_index = batch.text, batch.label, batch.follows_d, batch.follows_r, batch.tweet_index
        # decrease class labels by 1
        label.data.sub_(1)

        gts += list(label.data)
        tweet_indices += [int(m) for m in list(tweet_index.data)]
        # in case batch size smaller
        model.batch_size = int(label.data.shape[0])
        
        pred, batch_embeddings = model(text, follows_d, follows_r)

        for x in torch.exp(pred):
            softmax_scores[count_softmax] = x
            count_softmax += 1

        pred_label = pred.data.max(1)[1].cpu().numpy()
        preds += [x for x in pred_label]

        if save_embeddings:
            for x in batch_embeddings:
                sentence_embeddings[count] = x
                count += 1

        loss = loss_function(pred, label)
        avg_loss += float(loss)
    
    avg_loss /= len(data)
    acc = get_accuracy(gts, preds)

    # convert from list of torch Tensors to list of integers
    gts = [int(x.item()) for x in gts]

    if save_embeddings:
        return sentence_embeddings
    else:
        return avg_loss, acc, gts, preds, tweet_indices, softmax_scores

def write_results(tweet_indices, preds, softmax_scores):
    print("Saving results to CSV")
    write_csv_path = '../../results.csv'
    read_csv_path = '../../data.csv'

    df = pd.read_csv(read_csv_path, header=None, encoding='latin1')
    df.columns = [
        'tweet_id', 
        'user_id', 
        'retweet_user_id', 
        'text', 
        'party',
        'state', 
        'hashtags', 
        'keywords',
        'party_training',
        'index',
        'follows_d',
        'follows_r'
    ]

    assert len(df) == len(tweet_indices)

    results = []
    d_prob = []
    r_prob = []

    # convert to dict for O(1) processing
    tweet_indices_dict = {}
    for i, ind in enumerate(tweet_indices):
        tweet_indices_dict[ind] = i

    for i, row in tqdm(df.iterrows()):
        ind = tweet_indices_dict[row['index']]

        # use ground truth label if it exists
        if row['party'] != 'U':
            results.append(row['party'])
            if row['party'] == 'R':
                d_prob.append(0.0)
                r_prob.append(1.0)
            else:
                d_prob.append(1.0)
                r_prob.append(0.0)
        else:
            results.append('D') if int(preds[ind]) == 0 else results.append('R')
            d_prob.append(softmax_scores[ind][0])
            r_prob.append(softmax_scores[ind][1])

    assert len(results) == len(d_prob) == len(r_prob)

    df['d_prob'] = d_prob
    df['r_prob'] = r_prob
    df['results'] = results

    df.to_csv(path_or_buf=write_csv_path, header=False, index=False, encoding='latin1')

def run(args):

    N_SPLITS = 5

    five_fold_test_acc = []
    five_fold_test_acc_d = []
    five_fold_test_acc_r = []

    full_results = []

    if args.dataset == 'sentiment140':
        data_loader = Sentiment140Dataset(args, N_SPLITS)
        if args.text_cleaning: 
            csv_dir = '../../sentiment140_splits_cleaned'
        else:
            csv_dir = '../../sentiment140_splits'
    
    elif args.dataset == 'election2020':
        data_loader = Election2020Dataset(args, N_SPLITS)
        if args.save_embeddings: 
            csv_dir = '../../election2020_splits_embeddings'
        else:
            csv_dir = '../../election2020_splits'

    # Output path we write best model to
    output_path = make_output_path(args)
    
    # creates csv files which are then passed into the TabularDataset TorchText function call
    data_loader.gen_splits()
    
    # use full training set -- no 5-fold CV
    if args.full_training:
        N_SPLITS = 1
    
    for fold_n in range(N_SPLITS):
        best_model_file = os.path.join(output_path, 'fold{:02d}.pth'.format(fold_n))
        
        if args.model == 'bigru':
            text_field = data.Field(lower=True)
            label_field = data.Field(sequential=False)
            follows_d_field = data.Field(sequential=False)
            follows_r_field = data.Field(sequential=False)
            tweet_index_field = data.Field(sequential=False, use_vocab=False, dtype=torch.float)
        
            train_iter, val_iter, test_iter = load_iterators(
                text_field, 
                label_field, 
                follows_d_field,
                follows_r_field,
                tweet_index_field,
                args.batch_size, 
                fold_n, 
                csv_dir, 
                args.full_training, 
                args.save_embeddings,
                args.full_inference
            )

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
            
            if args.reweight_loss:
                prop_R = args.d_weight
                weight = torch.Tensor([prop_R, 1 - prop_R]).to(device)
            else:
                weight = torch.Tensor([1, 1]).to(device)
 
            loss_function = nn.NLLLoss(weight=weight)

            best_model = model 
            best_val_loss = float("inf")
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

                # ignore the validation and test sets for full training
                if args.full_training:
                    print()
                    continue

                with torch.no_grad():
                    val_loss, val_acc = val_evaluate(model, val_iter, loss_function)
                    print("Validation - Loss: {:.3f}, Accuracy: {:.2f}".format(val_loss, val_acc*100))
                    print()
                    
                    if val_loss < best_val_loss:
                        print("Validation loss improved from {:.5f} to {:.5f}. Saving model to {}".format(best_val_loss, val_loss, best_model_file))
                        
                        best_val_loss = val_loss
                        best_model = model

                        torch.save({
                            "epoch": epoch+1,
                            "best_val_loss": best_val_loss,
                            "model": best_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        }, best_model_file)

                        early_stop_idx = 0
                        reduce_lr_idx = 0
                    else:
                        print("Validation loss did not improve from {:.5f}".format(best_val_loss))
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

            with torch.no_grad():                
                if args.save_embeddings:
                    print("Saving embeddings...")
                    sentence_embeddings = test_evaluate(best_model, test_iter, loss_function, args.save_embeddings, args.hidden_dim)

                    # Save the embeddings
                    np.save(os.path.join(output_path, "fold{:02d}.npy".format(fold_n)), sentence_embeddings.cpu().numpy())
                else:
                    print("Evaluating Fold {}...".format(fold_n))
                    test_loss, test_acc, gts, preds, tweet_indices, softmax_scores = test_evaluate(best_model, test_iter, loss_function, args.save_embeddings, args.hidden_dim)
                    
                    print("Test Accuracy: {:.5f}".format(test_acc))
                    
                    matrix = confusion_matrix(gts, preds)
                    test_acc_d, test_acc_r = matrix.diagonal()/matrix.sum(axis=1)
                    
                    print("Test Accuracy-D: {:.5f}".format(test_acc_d))
                    print("Test Accuracy-R: {:.5f}".format(test_acc_r))
                    print("\n")

                    five_fold_test_acc.append(test_acc)
                    five_fold_test_acc_d.append(test_acc_d)
                    five_fold_test_acc_r.append(test_acc_r)

                    if args.full_inference:
                        write_results(list(tweet_indices), list(preds), softmax_scores)                          


    print("Final Results")
    print("------------------------------------")
    print("Test Accuracy: ", np.mean(np.array(five_fold_test_acc)))
    print("Test Accuracy-D: ", np.mean(np.array(five_fold_test_acc_d)))
    print("Test Accuracy-R: ", np.mean(np.array(five_fold_test_acc_r)))
    print("------------------------------------")

def read_args(args):
    parser = argparse.ArgumentParser(description=__doc__)

    # model selection parameters
    parser.add_argument('--model', dest='model', type=str, default='bigru', help='bigru')
    parser.add_argument('--dataset', dest='dataset', type=str, default='election2020', help='election2020,sentiment140')
    parser.add_argument('--seed', dest='seed', type=int, default=22)
    parser.add_argument('--output-dir', dest='output', type=str, default='results')
    parser.add_argument('--word2vec-path', dest='word2vec_path', type=str, default='../../GoogleNews-vectors-negative300.bin')
    parser.add_argument('--full-training', dest='full_training', action='store_true', help='Train on all available training data')
    parser.add_argument('--full-inference', dest='full_inference', action='store_true', help='Run inference on all available data (including unlabelled data')
    parser.add_argument('--save-embeddings', dest='save_embeddings', action='store_true', help='Save embeddings for GNN')

    # text parameters
    parser.add_argument('--text-cleaning', dest='text_cleaning', action='store_true', help='clean the data?')

    # model training parameters
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=128)
    parser.add_argument('--epochs', dest='epochs', type=int, default=2)
    parser.add_argument('--early-stopping', dest='early_stopping', type=int, default=1000)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3)
    parser.add_argument('--reduce-lr', dest='reduce_lr', type=int, default=1000)
    parser.add_argument('--num-workers', dest='num_workers', type=int, default=6)
    parser.add_argument('--d-weight', dest='d_weight', type=float, default=37657/122442)
    parser.add_argument('--reweight-loss', dest='reweight_loss', action='store_true', help='Reweight the loss function?')
    
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
    print("Dataset: ", args.dataset)
    print("Epochs", args.epochs)
    print("Learning Rate: ", args.lr) 
    print("Batch size: ", args.batch_size)
    print("Reweight Loss: ", args.reweight_loss)
    if args.reweight_loss:
        print("D Weight: ", args.d_weight)
    print()

    run(args)