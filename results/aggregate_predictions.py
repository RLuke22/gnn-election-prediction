import pandas as pd 
pd.set_option('display.max_colwidth',1000)
import os 

input_csv_path = "../../results_unweighted.csv"
df = pd.read_csv(input_csv_path, encoding='utf-8', header=None)

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
    'follows_r',
    'prob_d',
    'prob_r',
    'result'
]

# df = df.drop_duplicates(subset=['user_id'])
# df = df.drop_duplicates(subset=['text'])
# df = df.drop_duplicates(subset=['retweet_user_id'])
df = df.drop(['user_id', 'index', 'follows_d', 'follows_r'], axis=1)
df = df.drop(['hashtags', 'keywords'], axis=1)
df = df[df['retweet_user_id'] == -1]
df = df[df['state'] == 'Kansas']

df1 = df[df['prob_r'] >= 0.45]
df1 = df1[df1['prob_r'] <= 0.55]

df2 = pd.concat([df,df1]).drop_duplicates(subset=['tweet_id'], keep=False)

df = df2

# print(len(df))
# print(df[df.party_training == 'R'].shape[0])
# print(df[df.party_training == 'D'].shape[0])
# print(df[df.party == 'R'].shape[0])
# print(df[df.party == 'D'].shape[0])
print("D: ", df[df.result == 'D'].shape[0])
print("R: ", df[df.result == 'R'].shape[0])

