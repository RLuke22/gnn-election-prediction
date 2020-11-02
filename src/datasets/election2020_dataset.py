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
        self.csv_path = '../../election2020.csv'
        self.seed = args.seed
        self.model = args.model

        if self.text_cleaning:
            self.output_csv_dir = '../../election2020_splits_cleaned'
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
            'index'
        ]
        pd.options.display.max_columns = 2000
        # remove unnecessary rows
        df = df.drop(['user_id', 'retweet_user_id', 'party', 'state', 'hashtags', 'keywords', 'index'], axis=1)

        # replace R's with 0
        df['party_training'] = df['party_training'].replace('R',0)
        # replace D's with 1
        df['party_training'] = df['party_training'].replace('D',1)
        # Remove U's
        df = df[df.party_training != 'U']
        # remove nan rows
        df = df.dropna(how='any')

        # by suggestion from https://stackoverflow.com/a/47957914
        df = df.reset_index()
        # convert to lists
        text_data, party = list(df['text']), list(df['party_training'])

        # remove hashtags (except for #trump, #biden and so forth) and process emoticons
        text_data = self.data_preprocessor.twitter_clean(text_data)
        
        if self.text_cleaning:
            text_data = self.data_preprocessor.clean(text_data)
        
        li = []
        for text, label in zip(text_data, party):
            li.append((text, label))
        return li

    def gen_splits(self):

        # Five folds, each fold has a train, test, validation set
        if len(os.listdir(self.output_csv_dir)) == 3 * 5:
            return
        
        print("Generating CSV files...", end='')
        data_list = self.load_election2020()

        X = [str(tup[0]) for tup in data_list]
        y = [tup[1] for tup in data_list]

        print(len(X), len(y))

        # Use stratified K-fold to get most evenly-distributed dataset
        skf_test = StratifiedKFold(self.n_splits, random_state=self.seed, shuffle=True)

        for i, (train_val_index, test_index) in enumerate(skf_test.split(X, y)):            
            X_train_val, X_test = [X[j] for j in train_val_index], [X[j] for j in test_index]
            y_train_val, y_test = [y[j] for j in train_val_index], [y[j] for j in test_index]

            X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, stratify=y_train_val, train_size=0.75, shuffle=True, random_state=self.seed)
            
            if self.model == 'bigru':
                with open(os.path.join(self.output_csv_dir, 'train{:02d}.csv'.format(i)), 'w') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=['text', 'party'])
                    
                    for j in range(len(X_train)):
                        row = {
                            'text':X_train[j], 
                            'party':y_train[j]
                        }   
                        writer.writerow(row)

                with open(os.path.join(self.output_csv_dir, 'valid{:02d}.csv'.format(i)), 'w') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=['text', 'party'])
                    
                    for j in range(len(X_val)):
                        row = {
                            'text':X_val[j], 
                            'party':y_val[j]
                        }   
                        writer.writerow(row)
                
                with open(os.path.join(self.output_csv_dir, 'test{:02d}.csv'.format(i)), 'w') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=['text', 'party'])
                    
                    for j in range(len(X_test)):
                        row = {
                            'text':X_test[j], 
                            'party':y_test[j]
                        }   
                        writer.writerow(row)                

        print("Done.")




            







        

        






