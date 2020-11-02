import os 
import random
random.seed(22)

import pandas as pd
import pickle
import emoji
import csv
from sklearn.model_selection import StratifiedKFold, train_test_split

from datasets.preprocess import DataPreprocessor

class Election2020Dataset():
    def __init__(self, args, n_splits):
        super(Election2020Dataset, self).__init__()

        self.n_splits = n_splits
        self.text_cleaning = args.text_cleaning
        self.csv_path = '../../data.csv'
        self.seed = args.seed
        self.model = args.model
        self.full_training = args.full_training
        self.save_embeddings = args.save_embeddings

        if self.save_embeddings:
            self.output_csv_dir = '../../election2020_splits_embeddings'
        else:
            self.output_csv_dir = '../../election2020_splits'

        if not os.path.exists(self.output_csv_dir):
            os.mkdir(self.output_csv_dir)

        self.data_preprocessor = DataPreprocessor()

    def load_election2020(self):

        df = pd.read_csv(self.csv_path, header=None, encoding='utf-8')
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
        pd.options.display.max_columns = 2000
        # remove unnecessary rows
        df = df.drop(['user_id', 'retweet_user_id', 'party', 'state', 'hashtags', 'keywords', 'index'], axis=1)

        # replace R's with 0
        df['party_training'] = df['party_training'].replace('R', 0)
        # replace D's with 1
        df['party_training'] = df['party_training'].replace('D', 1)

        if not self.save_embeddings:
            # Remove U's
            df = df[df.party_training != 'U']

        # Just set to 1 -- either not going to be any 'U's or we are only getting embeddings
        df['party_training'] = df['party_training'].replace('U', 1)

        # convert to lists
        text_data, party, follows_d, follows_r = list(df['text']), list(df['party_training']), list(df['follows_d']), list(df['follows_r'])

        # remove hashtags (except for #trump, #biden and so forth) and process emoticons
        text_data = self.data_preprocessor.twitter_clean(text_data)
        
        if self.text_cleaning:
            text_data = self.data_preprocessor.clean(text_data)
        
        li = []
        for text, label, one_hot_d, one_hot_r in zip(text_data, party, follows_d, follows_r):
            li.append((text, label, one_hot_d, one_hot_r))
        return li

    def gen_splits(self):

        # Five folds, each fold has a train, test, validation set
        if len(os.listdir(self.output_csv_dir)) == 3 * 5 + 1:
            return
        
        print("Generating CSV files...", end='')
        data_list = self.load_election2020()

        X = [str(tup[0]) for tup in data_list]
        y = [tup[1] for tup in data_list]
        d = [tup[2] for tup in data_list]
        r = [tup[3] for tup in data_list]

        print("Data sizes:")
        print(len(X), len(y), len(d), len(r))

        print("Constructing full training set...", end="")
        with open(os.path.join(self.output_csv_dir, 'full_data.csv'), 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=['text', 'party', 'follows_d', 'follows_r'])
            
            for j in range(len(X)):
                row = {
                    'text':X[j], 
                    'party':y[j],
                    'follows_d': d[j],
                    'follows_r': r[j]
                }   
                writer.writerow(row)

        print("Constructing 5-fold splits...", end="")
        # Use stratified K-fold to get most evenly-distributed dataset
        skf_test = StratifiedKFold(self.n_splits, random_state=self.seed, shuffle=True)

        for i, (train_val_index, test_index) in enumerate(skf_test.split(X, y)):            
            X_train_val, X_test = [X[j] for j in train_val_index], [X[j] for j in test_index]
            y_train_val, y_test = [y[j] for j in train_val_index], [y[j] for j in test_index]
            d_train_val, d_test = [d[j] for j in train_val_index], [d[j] for j in test_index]
            r_train_val, r_test = [r[j] for j in train_val_index], [r[j] for j in test_index]

            features_train_val = []
            for text_features, one_hot_d, one_hot_r in zip(X_train_val, d_train_val, r_train_val):
                features_train_val.append((text_features, one_hot_d, one_hot_r))

            features_train, features_val, y_train, y_val = train_test_split(features_train_val, y_train_val, stratify=y_train_val, train_size=0.75, shuffle=True, random_state=self.seed)
            X_train, X_val = [feature[0] for feature in features_train], [feature[0] for feature in features_val]
            d_train, d_val = [feature[1] for feature in features_train], [feature[1] for feature in features_val]
            r_train, r_val = [feature[2] for feature in features_train], [feature[2] for feature in features_val]

            print("Split sizes:")
            print(len(X_train), len(X_val), len(X_test))
            print(len(y_train), len(y_val), len(y_test))
            print(len(d_train), len(d_val), len(d_test))
            print(len(r_train), len(r_val), len(r_test))
            
            with open(os.path.join(self.output_csv_dir, 'train{:02d}.csv'.format(i)), 'w') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=['text', 'party', 'follows_d', 'follows_r'])
                
                for j in range(len(X_train)):
                    row = {
                        'text':X_train[j], 
                        'party':y_train[j],
                        'follows_d':d_train[j],
                        'follows_r':r_train[j]
                    }   
                    writer.writerow(row)

            with open(os.path.join(self.output_csv_dir, 'valid{:02d}.csv'.format(i)), 'w') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=['text', 'party', 'follows_d', 'follows_r'])
                
                for j in range(len(X_val)):
                    row = {
                        'text':X_val[j], 
                        'party':y_val[j],
                        'follows_d':d_val[j],
                        'follows_r':r_val[j]
                    }   
                    writer.writerow(row)
            
            with open(os.path.join(self.output_csv_dir, 'test{:02d}.csv'.format(i)), 'w') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=['text', 'party', 'follows_d', 'follows_r'])
                
                for j in range(len(X_test)):
                    row = {
                        'text':X_test[j], 
                        'party':y_test[j],
                        'follows_d':d_test[j],
                        'follows_r':r_test[j]
                    }   
                    writer.writerow(row)                

        print("Done.")




            







        

        






