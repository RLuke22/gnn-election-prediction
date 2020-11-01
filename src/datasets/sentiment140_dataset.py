import numpy as np 
import os 
import random
random.seed(22)
from numpy.random import seed
seed(22)

import pandas as pd
import csv
from sklearn.model_selection import StratifiedKFold, train_test_split

from datasets.preprocess import DataPreprocessor

class Sentiment140Dataset():
    def __init__(self, args, n_splits):
        super(Sentiment140Dataset, self).__init__()

        self.n_splits = n_splits
        self.text_cleaning = args.text_cleaning
        self.csv_path = '../../sentiment140.csv'
        self.seed = args.seed

        if self.text_cleaning:
            self.output_csv_dir = '../../sentiment140_splits_cleaned'
        else:
            self.output_csv_dir = '../../sentiment140_splits'

        if not os.path.exists(self.output_csv_dir):
            os.mkdir(self.output_csv_dir)

        self.data_preprocessor = DataPreprocessor()

    def load_sentiment140(self):

        df = pd.read_csv(self.csv_path, header=None, encoding='latin1')
        # only classify with text and sentiment
        df.columns = ['sentiment', 'id', 'date', 'query', 'user_id', 'text']
        df = df.drop(['id', 'date', 'query', 'user_id'], axis=1)
        # replace 4's (positive) with 1
        df['sentiment'] = df['sentiment'].replace(4,1)

        # convert to lists
        text_data, sentiment = list(df['text']), list(df['sentiment'])

        if self.text_cleaning:
            text_data = self.data_preprocessor.clean(text_data)
        
        li = []
        for text, label in zip(text_data, sentiment):
            li.append((text, label))

        return li

    def gen_splits(self):

        # Five folds, each fold has a train, test, validation set
        if len(os.listdir(self.output_csv_dir)) == 3 * 5:
            return
        
        print("Generating CSV files...", end='')
        data_list = self.load_sentiment140()

        X = [tup[0] for tup in data_list]
        y = [tup[1] for tup in data_list]

        # Use stratified K-fold to get most evenly-distributed dataset
        skf_test = StratifiedKFold(self.n_splits, random_state=self.seed, shuffle=True)

        for i, (train_val_index, test_index) in enumerate(skf_test.split(X, y)):            
            X_train_val, X_test = [X[j] for j in train_val_index], [X[j] for j in test_index]
            y_train_val, y_test = [y[j] for j in train_val_index], [y[j] for j in test_index]

            X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, stratify=y_train_val, train_size=0.75, shuffle=True, random_state=self.seed)
            
            with open(os.path.join(self.output_csv_dir, 'train{:02d}.csv'.format(i)), 'w') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=['text', 'sentiment'])
                
                for j in range(len(X_train)):
                    row = {
                        'text':X_train[j], 
                        'sentiment':y_train[j]
                    }   
                    writer.writerow(row)

            with open(os.path.join(self.output_csv_dir, 'valid{:02d}.csv'.format(i)), 'w') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=['text', 'sentiment'])
                
                for j in range(len(X_val)):
                    row = {
                        'text':X_val[j], 
                        'sentiment':y_val[j]
                    }   
                    writer.writerow(row)
            
            with open(os.path.join(self.output_csv_dir, 'test{:02d}.csv'.format(i)), 'w') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=['text', 'sentiment'])
                
                for j in range(len(X_test)):
                    row = {
                        'text':X_test[j], 
                        'sentiment':y_test[j]
                    }   
                    writer.writerow(row)

        print("Done.")




            







        

        






