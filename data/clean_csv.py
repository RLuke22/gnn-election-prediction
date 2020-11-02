import pandas as pd 
import os
from tqdm import tqdm
import numpy as np
import pickle

input_csv_path = '../../election2020_raw.csv'
output_csv_path = '../../election2020.csv'

# df = pd.read_csv(input_csv_path, header=None, encoding='latin1')
# df.columns = [
#     'tweet_id', 
#     'user_id', 
#     'retweet_user_id', 
#     'text', 
#     'party',
#     'state', 
#     'hashtags', 
#     'keywords'
# ]

# # Remove duplicate tweet_ids (keeps the first)
# df = df.drop_duplicates(subset=['tweet_id'])

# # Table without duplicates
# df1 = df.drop_duplicates(subset=['text'])
# df2 = pd.concat([df,df1]).drop_duplicates(subset=['tweet_id'], keep=False)

# # df2 contains all the removed duplicates
# assert len(df1) + len(df2) == len(df)

# party_training = []
# for index, row in tqdm(df.iterrows()):
#     if row['tweet_id'] in df1.tweet_id.values:
#         party_training.append(row['party'])
#     else:
#         # Everything in the duplicates DataFrame become 'U' to not affect training
#         party_training.append('U')

# # New column in dataframe
# df['party_training'] = party_training

# # Shuffle DataFrame
# df = df.sample(frac=1).reset_index(drop=True)
# df['index'] = np.arange(len(df))

# df.to_csv(path_or_buf=output_csv_path, header=False, index=False, encoding='latin1')

# index_to_tweetid = {}
# tweetid_to_index = {}
# for index, row in tqdm(df.iterrows()):
#     index_to_tweetid[row['index']] = row['tweet_id']
#     tweetid_to_index[row['tweet_id']] = row['index']

# with open('../../index_to_tweetid.pkl', 'wb') as f:
#     pickle.dump(index_to_tweetid, f)

# with open('../../tweetid_to_index.pkl', 'wb') as f:
#     pickle.dump(tweetid_to_index, f)

df = pd.read_csv(self.csv_path, header=None, encoding='utf-8')


    





