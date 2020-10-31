import sys
import os 
import pymongo 
import tweepy

from tqdm import tqdm

client = pymongo.MongoClient("mongodb+srv://Quintillion:TzjTGcE5I6Bu7P9e@twitterdata.wkwqp.mongodb.net/TwitterData?retryWrites=true&w=majority")
db = client['TwitterData']
tweets_collection = db['Tweets']
followers_collection = db['Followers']

cursor = tweets_collection.aggregate(
        [ 
            { '$sample': { 'size': 50 } }
        ]
    )

for sample in tqdm(cursor):
    followers_collection.insert_one(sample)

print("Done.")
