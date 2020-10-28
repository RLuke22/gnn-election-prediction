import argparse
import sys 
import os 

import pymongo
import tweepy
import numpy as np

def read_args(args):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--user', dest='user', type=str, default='lrowe', help='lrowe or qyong')
    parser.add_argument('--end-time', dest='end_time', type=str, default='2020-11-03', help='YYYY-MM-DD')

    return parser.parse_args(args)

class TweetDataEngine():
    def __init__(self, args):
        super(TweetDataEngine, self).__init__()

        self.user = args.user
        self.end_time = args.end_time

        self.client = pymongo.MongoClient("mongodb+srv://Quintillion:TzjTGcE5I6Bu7P9e@twitterdata.wkwqp.mongodb.net/TwitterData?retryWrites=true&w=majority")
        self.db = self.client['TwitterData']

        if self.user == 'lrowe':
            self.consumer_key_Q = "JZNzXeOD8VMDCroiKFXsXwAdg"
            self.consumer_secret_Q = "O58HHCidMQ8bogw4ofs8hr50V45aYxAG2i9vvqfZBFPXI3zKjM"

            self.backup_file = 'lrowe_data.csv'
        elif self.user == 'qyong':
            self.consumer_key_Q = "DT3XJxLbG9gSAExLTnw8Ya8wy"
            self.consumer_secret_Q = "9zhg6sRhqaBBgzYyBuoaohPH6WLpYEMWkg3S0koAhr50dfZQ1q"

            self.backup_file = 'qyong_data.csv' 

        if not os.path.exists(self.backup_file):
            os.mkdir(self.backup_file)

        self.api = tweepy.API(tweepy.AppAuthHandler(self.consumer_key_Q, self.consumer_secret_Q))

        self.hashtags_d_leaning = ['#biden2020', '#bidenharris2020', '#votebidenharris2020', '#theresistance', '#resist', '#votebluetosaveamerica',
                        '#votebiden', '#voteblue', '#bidenharrislandslide2020', '#votehimout', '#dumptrump']
        self.hashtags_r_leaning = hashtags_r_leaning = ['#trump2020', '#trumppence2020', '#maga', '#maga2020', '#kag', '#voteredtosaveamerica', '#votetrump', 
                        '#votered', '#trump2020landslide', '#sleepyjoe', '#neverbiden']
        self.hashtags_n_leaning = ['#biden', '#bidenharris', '#joebiden', '#kamalaharris', '#trump', '#trumppence', '#donaldtrump', '#mikepence']
    
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
                'Texas0': '27.7193,-97.6058.50mi,124.43mi',
                'Texas1': '30.8456,-96.8243 ,187.81mi',
                'Texas2': '31.8902,-106.4866,8.45mi',
                'Texas3': '32.4299,-100.4680,147.90mi',
                'NorthCarolina0': '35.3129,-78.2647,84.16mi',
                'NorthCarolina1': '35.6695,-80.1571,58.10mi',
                'NorthCarolina2': '35.7695,-81.5075,38.31mi',
                'NorthCarolina3': '35.5960,-82.4469,26.29mi'
            }

        self.region_tweet_counts = {
                'Arizona0': 5818, 
                'Florida0':16450,
                'Florida1':889,
                'Iowa0': 290, 
                'Iowa1': 1812, 
                'Iowa2': 246, 
                'Iowa3': 157, 
                'Georgia0': 6304, 
                'Georgia1': 2164,
                'Ohio0': 994,
                'Ohio1': 4182,
                'Ohio2': 651,
                'Ohio3': 3435,
                'Texas0': 2997,
                'Texas1': 18262,
                'Texas2': 660,
                'Texas3': 1315,
                'NorthCarolina0': 3773,
                'NorthCarolina1': 3172,
                'NorthCarolina2': 979,
                'NorthCarolina3': 445
        }

    def query(self):
        hashtags = self.hashtags_d_leaning + self.hashtags_r_leaning + self.hashtags_n_leaning
        return ' OR '.join(hashtags)

    def extract_data(self):
        
        return None

if __name__ == '__main__':
    args = read_args(sys.argv[1:])

    print("Polling configurations")
    print("_____________________")
    print("User: ", args.user)
    print("End Time: ", args.end_time)
    print()

    data_engine = TweetDataEngine(args)
    data_engine.extract_data()

    
