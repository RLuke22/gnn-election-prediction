import torch
import os
import pickle
from torch_geometric.data import Data
from tqdm import tqdm
from pandas import read_csv
import numpy as np

class GNNDataset():
    def __init__(self, args):
        super(GNNDataset, self).__init__()

        self.csv_path = '../../data.csv'
        self.d_edge_list_path = '../../d_edge_list.pkl'
        self.r_edge_list_path = '../../r_edge_list.pkl'
        self.train_val_test_path = '../../election2020_splits'
        self.gru_output_dir = args.gru_output_dir       

    def create_gnn_dataset(self, fold_n):        
        df = read_csv(self.csv_path, header=None, encoding='latin1')
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

        # We first index the tweets from 0 - 655026 as the edge index 
        # is defined over this index
        tweet_indices = list(df['index'])
        new_tweet_indices = {}
        for i, index in enumerate(tweet_indices):
            new_tweet_indices[index] = i 

        # Let us first create the edge list
        with open(self.d_edge_list_path, 'rb') as f:
            d_edge_list_tuples = pickle.load(f)
        with open(self.r_edge_list_path, 'rb') as f:
            r_edge_list_tuples = pickle.load(f)
        edge_list_tuples = d_edge_list_tuples + r_edge_list_tuples
        
        edge_index = torch.empty((2, len(edge_list_tuples)), dtype=torch.long)
        for i, (node1, node2) in tqdm(enumerate(edge_list_tuples)):
            edge_index[0, i] = new_tweet_indices[node1]
            edge_index[1, i] = new_tweet_indices[node2]

        # We now create the feature matrices
        # The feature matrices are already ordered in increasing order of index, so no processing required here
        features = torch.from_numpy(np.load(os.path.join(self.gru_output_dir, "fold{:02d}_embeddings.npy".format(fold_n))))

        # Let us now create the label matrix
        # replace R's with 0
        df['party_training'] = df['party_training'].replace('R', 0)
        # replace D's with 1
        df['party_training'] = df['party_training'].replace('D', 1)
        # replace U's with 2 (won't be used in training)
        df['party_training'] = df['party_training'].replace('U', 2)

        labels = torch.Tensor(list(df['party_training']))
        labels = labels.type(torch.LongTensor)
        # We now create the masks for train/validation/test
        train_csv_path = os.path.join(self.train_val_test_path, 'train{:02d}.csv'.format(fold_n))
        val_csv_path = os.path.join(self.train_val_test_path, 'valid{:02d}.csv'.format(fold_n))
        test_csv_path = os.path.join(self.train_val_test_path, 'test{:02d}.csv'.format(fold_n))

        train_test_val_columns = [
            'text', 
            'party',
            'follows_d',
            'follows_r',
            'tweet_index'
        ]

        df_train = read_csv(train_csv_path, header=None, encoding='latin1')
        df_val = read_csv(val_csv_path, header=None, encoding='latin1')
        df_test = read_csv(test_csv_path, header=None, encoding='latin1')

        df_train.columns = train_test_val_columns
        df_val.columns = train_test_val_columns
        df_test.columns = train_test_val_columns

        train_mask = torch.zeros(len(df), dtype=torch.bool)
        val_mask = torch.zeros(len(df), dtype=torch.bool)
        test_mask = torch.zeros(len(df), dtype=torch.bool)
        test_d_mask = torch.zeros(len(df), dtype=torch.bool)
        test_r_mask = torch.zeros(len(df), dtype=torch.bool)

        for index in list(df_train['tweet_index']):
            train_mask[new_tweet_indices[index]] = True
        for index in list(df_val['tweet_index']):
            val_mask[new_tweet_indices[index]] = True
        for index in list(df_test['tweet_index']):
            test_mask[new_tweet_indices[index]] = True
        for _, row in df_test.iterrows():
            if row['party'] == 0:
                test_r_mask[new_tweet_indices[row['tweet_index']]] = True
            elif row['party'] == 1:
                test_d_mask[new_tweet_indices[row['tweet_index']]] = True


        return Data(x=features, y=labels, edge_index=edge_index, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask), test_d_mask, test_r_mask

        
