import tweepy
import csv 
import os 
import sys 
import argparse
import time
import random
from tqdm import tqdm 

class TwitterFollowerDataEngine():
    def __init__(self, args):
        super(TwitterFollowerDataEngine, self).__init__()

        self.user = args.user 
        
        if self.user == 'lrowe':
            self.consumer_key_Q = "JZNzXeOD8VMDCroiKFXsXwAdg"
            self.consumer_secret_Q = "O58HHCidMQ8bogw4ofs8hr50V45aYxAG2i9vvqfZBFPXI3zKjM"

            self.backup_file = '../../lrowe_followers.csv'

            self.twitter_accounts = {
                'Mike_Pence': 'R',
                'seanhannity': 'R',
                'TeamTrump': 'R',
                'NRA': 'R'
            }
        elif self.user == 'qyong':
            self.consumer_key_Q = "DT3XJxLbG9gSAExLTnw8Ya8wy"
            self.consumer_secret_Q = "9zhg6sRhqaBBgzYyBuoaohPH6WLpYEMWkg3S0koAhr50dfZQ1q"

            self.backup_file = '../../qyong_followers.csv' 

            self.twitter_accounts = {
                'JoeBiden': 'D',
                'TuckerCarlson': 'R'
            }
        
        self.api = tweepy.API(tweepy.AppAuthHandler(self.consumer_key_Q, self.consumer_secret_Q))
        self.fieldnames = ['user_id', 'political_leaning']

    def extract_followers(self):

        print("Extracting followers...")
        count = 0

        for t_account in self.twitter_accounts.keys():
            for page in tweepy.Cursor(self.api.followers_ids, screen_name=t_account).pages():
                time.sleep(60)

                with open(self.backup_file, 'a+') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)                
                    
                    # this for loop runs in about 0.01 seconds (not a bottleneck)
                    for user_id in page:
                        count += 1
                        # print roughly every 5 minutes
                        if count % 25000 == 0:
                            print("Extracted {} followers!".format(count))

                        writer.writerow({
                            'user_id': user_id,
                            'political_leaning': self.twitter_accounts[t_account]
                        })
        
        print("Done!")

        # TEST
        # t_account = 'Mike_Pence'
        # with open(self.backup_file, 'a+') as csvfile:
        #     writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)     
            
        #     page = [random.randint(0, 100000000000) for iter in range(5000)]
            
        #     start = time.time()
        #     for user_id in page:
        #         writer.writerow({
        #             'user_id': user_id,
        #             'political_leaning': self.twitter_accounts[t_account]
        #         })
        #     end = time.time()
        #     print("{}s".format(end - start))
        #     print("Done.")

def read_args(args):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--user', dest='user', type=str, default='lrowe', help='lrowe or qyong')
    
    return parser.parse_args(args)

if __name__ == '__main__':
    args = read_args(sys.argv[1:])

    print("Polling configurations")
    print("_____________________")
    print("User: ", args.user)
    print()

    data_engine = TwitterFollowerDataEngine(args)
    data_engine.extract_followers()