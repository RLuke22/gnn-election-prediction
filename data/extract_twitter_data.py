import argparse
import sys 
import os 
import re
from datetime import datetime
import csv

import pymongo
import tweepy

from tqdm import tqdm

class TweetDataEngine():
    def __init__(self, args):
        super(TweetDataEngine, self).__init__()

        self.user = args.user
        self.start_time = args.start_time
        self.end_time = args.end_time
        self.hashtag_list = args.hashtag_list

        self.client = pymongo.MongoClient("mongodb+srv://Quintillion:TzjTGcE5I6Bu7P9e@twitterdata.wkwqp.mongodb.net/TwitterData?retryWrites=true&w=majority")
        self.db = self.client['TwitterData']

        # collection = self.db['Tweets']
        # c = collection.distinct("user_id",{"party": "R", "state": "Georgia"})
        # print(len(c))
        # exit(0)

        if self.user == 'lrowe':
            self.consumer_key_Q = "JZNzXeOD8VMDCroiKFXsXwAdg"
            self.consumer_secret_Q = "O58HHCidMQ8bogw4ofs8hr50V45aYxAG2i9vvqfZBFPXI3zKjM"

            self.backup_file = '../../lrowe_data.csv'
        elif self.user == 'qyong':
            self.consumer_key_Q = "DT3XJxLbG9gSAExLTnw8Ya8wy"
            self.consumer_secret_Q = "9zhg6sRhqaBBgzYyBuoaohPH6WLpYEMWkg3S0koAhr50dfZQ1q"

            self.backup_file = '../../qyong_data.csv' 

        self.fieldnames = ['tweet_id', 'user_id', 'text', 'state', 'party', 'in_graph', 'retweet_user_id']
        self.api = tweepy.API(tweepy.AppAuthHandler(self.consumer_key_Q, self.consumer_secret_Q), wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
        
        if self.hashtag_list == 1:
            self.hashtags_d_leaning = [ 
                '#votebidenharris2020', 
                '#theresistance', 
                '#resist', 
                '#votebiden', 
                '#voteblue', 
                '#votehimout', 
                '#dumptrump'
            ]
            self.hashtags_r_leaning = [ 
                '#maga', 
                '#maga2020', 
                '#kag', 
                '#votetrump', 
                '#votered', 
                '#sleepyjoe', 
                '#neverbiden'
            ]
            self.hashtags_n_leaning = [
                '#trump2020', 
                '#trumppence2020',
                '#biden2020', 
                '#bidenharris2020',
                '#biden', 
                '#bidenharris', 
                '#joebiden', 
                '#kamalaharris', 
                '#trump', 
                '#trumppence', 
                '#donaldtrump', 
                '#mikepence'
            ]
        elif self.hashtag_list == 2:
            self.hashtags_d_leaning = [
                '#votebluetosaveamerica',
                '#votebidenharris',
                '#votebluetoendthisnightmare',
                '#voteblue2020',
                '#bidenharrislandslide2020',
                '#buildbackbetter',
                '#votebidenharristosaveamerica',
            ]
            self.hashtags_r_leaning = [
                '#voteredtosaveamerica',
                '#votetrumppence',
                '#fourmoreyears',
                '#votered2020',
                '#trump2020landslide',
                '#magaa',
                '#keepamericagreat'
            ]
            self.hashtags_n_leaning = [
                '#democrats',
                '#democrat',
                '#republicans',
                '#republican'
            ]
        elif self.hashtag_list == 3:
            self.hashtags_d_leaning = [
                '#votebluenomatterwho',
                '#bidenharris2020landslide',
                '#bluewave',
                '#bluewave2020',
                '#bidenharristosaveamerica',
                '#joebiden2020',
                '#votehimout2020',
            ]
            self.hashtags_r_leaning = [
                '#trump2020nowmorethanever',
                '#donaldtrump2020',
                '#makeamericagreatagain',
                '#kag2020',
                '#trumptrain',
                '#voteredtosaveamerica2020',
                '#trump2020tosaveamerica'
            ]
            self.hashtags_n_leaning = [
                '#hunterbiden',
                '#teamtrump',
                '#teambiden',
                '#americafirst'
            ]

        elif self.hashtag_list == 4:
            self.hashtags_d_leaning = [
            ]
            self.hashtags_r_leaning = [
            ]
            self.hashtags_n_leaning = [
                'Biden',
                'Trump',
                'Democrat',
                'Republican',
                'Democrats',
                'Republicans'
            ]

        self.hashtags = self.hashtags_d_leaning + self.hashtags_r_leaning + self.hashtags_n_leaning
        
        self.geocodes = {
                'Arizona0': '33.8244,-111.5818,146.4mi', 
                'Florida0':'27.1984,-83.0723,251.73mi',
                'Florida1':'29.4065,-86.1746,114mi',
                'Iowa0': '42.7802,-95.5281,51.26mi', 
                'Iowa1': '42.0493,-92.8059,100.05mi', 
                'Iowa2': '41.87931,-90.7568,27.67mi', 
                'Iowa3': '41.2540,-95.1875,37.76mi', 
                'Georgia0': '33.5816,-83.9150,78.99mi', 
                'Georgia1': '32.1667,-82.7974,109.92mi',
                'Ohio0': '39.1930,-84.6620,8.81mi',
                'Ohio1': '39.8454,-83.3036,79.47mi',
                'Ohio2': '41.2866,-83.3382,34.05mi',
                'Ohio3': '41.3320,-81.6402,56.99mi',
                'Texas0': '27.7193,-97.6058,124.43mi',
                'Texas1': '30.8456,-96.8243,187.81mi',
                'Texas2': '31.8902,-106.4866,8.45mi',
                'Texas3': '32.4299,-100.4680,147.90mi',
                'NorthCarolina0': '35.3129,-78.2647,84.16mi',
                'NorthCarolina1': '35.6695,-80.1571,58.10mi',
                'NorthCarolina2': '35.7695,-81.5075,38.31mi',
                'NorthCarolina3': '35.5960,-82.4469,26.29mi'
            }

        # 1000000 tweet cap
        # Thus, we have ~75000 tweets per day
        # Thus, we have ~37500 tweets per hashtag list per day
        # The 37500 retrieved tweets are distributed according to each region's population (manually calculated using 2020 Census data)
        self.region_tweet_counts = {
                'Arizona0': 2909, 
                'Florida0':8225,
                'Florida1':445,
                'Iowa0': 145, 
                'Iowa1': 906, 
                'Iowa2': 123, 
                'Iowa3': 79, 
                'Georgia0': 3152, 
                'Georgia1': 1082,
                'Ohio0': 497,
                'Ohio1': 2091,
                'Ohio2': 326,
                'Ohio3': 1718,
                'Texas0': 1499,
                'Texas1': 9131,
                'Texas2': 330,
                'Texas3': 658,
                'NorthCarolina0': 1887,
                'NorthCarolina1': 1586,
                'NorthCarolina2': 490,
                'NorthCarolina3': 223
        }

    def query(self):
        return ' OR '.join(self.hashtags)

    def time_exceeded(self, tweet_timestamp):
        start_year = int(self.start_time[0:4])
        start_month = int(self.start_time[5:7])
        start_day = int(self.start_time[8:10])

        start_timestamp = datetime(start_year, start_month, start_day)

        if tweet_timestamp < start_timestamp:
            #print("\nTweet posted at {}, which is before {}. Early stopping...".format(str(tweet_timestamp), str(start_timestamp)))
            return True
        else:
            return False

    def get_party(self, tweet):
        tweet_hashtags = []
        for ht in self.hashtags:
            if ht in tweet.lower():
                tweet_hashtags.append(ht)

        d_hashes = [h for h in tweet_hashtags if h.lower() in self.hashtags_d_leaning]
        r_hashes = [h for h in tweet_hashtags if h.lower() in self.hashtags_r_leaning]
        n_hashes = [h for h in tweet_hashtags if h.lower() in self.hashtags_n_leaning]

        if d_hashes and r_hashes:
            party = 'U'
        elif d_hashes:
            party = 'D'
        elif r_hashes:
            party = 'R'
        else:
            party = 'U'

        return party
        

    def extract_data(self):
        collection = self.db['Tweets']

        print("Extracting tweets...")
        for region in self.geocodes.keys():
            tweet_count = self.region_tweet_counts[region]
            # for logging

            for i, tweet_info in enumerate(tweepy.Cursor(self.api.search, 
                                            q=self.query(), 
                                            count=tweet_count, 
                                            result_type="recent", 
                                            tweet_mode="extended", 
                                            geocode=self.geocodes[region], 
                                            lang='en',
                                            until=self.end_time
                                            ).items(tweet_count)):

                # # for logging
                # print('{}-'.format(i), end='')

                # early stopping checks
                # tweets retrieved in reverse chronological order
                if self.time_exceeded(tweet_info.created_at):
                    break
                # check if already in database
                if collection.find_one({"tweet_id" : tweet_info.id}):
                    continue

                if 'retweeted_status' in dir(tweet_info):
                    tweet=tweet_info.retweeted_status.full_text
                    original_tweet_user_id = tweet_info.retweeted_status.user.id
                else:
                    tweet=tweet_info.full_text
                    original_tweet_user_id = -1

                # remove unnecessary URLs
                tweet = re.sub(r"http\S+", "", tweet)
                party = self.get_party(tweet)

                row = {
                    "tweet_id":tweet_info.id, 
                    "user_id":tweet_info.user.id, 
                    "text":tweet, 
                    "state":region[:-1], 
                    "party":party, 
                    "in_graph":0,
                    "retweet_user_id": original_tweet_user_id
                }
                with open(self.backup_file, 'a+') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                    writer.writerow(row)
                
                collection.insert_one(row)
            
            print('Region: {}, Tweets: {}'.format(region,i))

def read_args(args):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--user', dest='user', type=str, default='lrowe', help='lrowe or qyong')
    parser.add_argument('--end-time', dest='end_time', type=str, default='2020-11-03', help='YYYY-MM-DD')
    parser.add_argument('--start-time', dest='start_time', type=str, default='2020-11-02', help='YYYY-MM-DD')
    parser.add_argument('--hashtag-list', dest='hashtag_list', type=int, default=1, help='1,2')
    
    return parser.parse_args(args)

if __name__ == '__main__':
    args = read_args(sys.argv[1:])

    print("Polling configurations")
    print("_____________________")
    print("User: ", args.user)
    print("Start Time: ", args.start_time)
    print("End Time: ", args.end_time)
    print("Hashtag List: ", args.hashtag_list)
    print()

    data_engine = TweetDataEngine(args)
    data_engine.extract_data()
