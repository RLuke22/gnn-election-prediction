import csv
import os
import sys
import argparse
import pickle
import numpy as np
import random

class FollowerDictsEngine():
    def __init__(self, args):
        super(FollowerDictsEngine, self).__init__()

        self.follower_files = args.follower_files
        self.tweet_index_dict_file = args.tweet_index_dict_file
        self.tweet_table_csv = args.tweet_table_csv
        self.user_tweet_dict_file = args.user_tweet_dict_file
        self.d_list_file = args.d_list
        self.r_list_file = args.r_list

    def create_userid_to_tweetids_dict(self):
        userid_to_tweetids = {}
        skipped = 0

        with open(self.tweet_table_csv, newline='') as f:
            csv_reader = csv.reader(f, delimiter=',')

            print_every_n = 10000
            for i,row in enumerate(csv_reader):
                if row[0] == '':
                    skipped += 1
                    continue

                tweetid = int(row[0])
                userid  = int(row[1])                

                if userid not in userid_to_tweetids.keys():
                    userid_to_tweetids[userid] = [tweetid]
                else:
                    userid_to_tweetids[userid].append(tweetid)

                if i % print_every_n == 0:
                    print("Added " + str(i) + " to TweetID UserID dict.")

        with open('userid_to_tweetids.pkl', 'wb') as handle:
            pickle.dump(userid_to_tweetids, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(skipped)

    def create_dr_tweet_list(self):
        d_list = []
        r_list = []

        with open(self.tweet_index_dict_file, 'rb') as handle:
            tweet_index = pickle.load(handle)
        
        with open(self.user_tweet_dict_file, 'rb') as handle:
            userid_to_tweetids = pickle.load(handle)

        rows_written = 0
        print_every_n = 100000
        for follower_file in self.follower_files:
            print(follower_file)
            with open(follower_file, newline='') as f:
                csv_reader = csv.reader(f, delimiter=',')

                for row in csv_reader:
                    userid = int(row[0])
                    party  = row[1]

                    if userid in userid_to_tweetids.keys():
                        for tweetid in userid_to_tweetids[userid]:
                            index = tweet_index[tweetid]
                            if party == 'D':
                                d_list.append(index)
                            elif party == 'R':
                                r_list.append(index)

                    rows_written += 1
                    if rows_written % print_every_n == 0:
                        print(str(rows_written) + " rows written")
                        print("Removing duplicates")
                        d_list = list(dict.fromkeys(d_list))
                        r_list = list(dict.fromkeys(r_list))

        print(str(rows_written) + " rows written")       
        print("Removing duplicates")
        d_list = list(dict.fromkeys(d_list))
        r_list = list(dict.fromkeys(r_list))

        with open('d_followers_tweets.pkl', 'wb') as handle:
            pickle.dump(d_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open('r_followers_tweets.pkl', 'wb') as handle:
            pickle.dump(r_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def create_edge_list(self):
        with open(self.d_list_file, 'rb') as handle:
            d_list = pickle.load(handle)

        with open(self.r_list_file, 'rb') as handle:
            r_list = pickle.load(handle)

        n_rand_neighbours = 100
        print_every_n = 5000000

        print("Constructing D edges: " + str(len(d_list) * n_rand_neighbours * 2))
        d_edges = []
        d_edges_created = 0
        for u in d_list:
            rand_neighbours = random.sample(d_list, n_rand_neighbours)
            while u in rand_neighbours:
                rand_neighbours = random.sample(d_list, n_rand_neighbours)

            for v in rand_neighbours:
                d_edges.append((u, v))
                d_edges.append((v, u))
            d_edges_created += n_rand_neighbours

            if d_edges_created % print_every_n == 0:
                print(d_edges_created)

        with open('d_edge_list.pkl', 'wb') as handle:
            pickle.dump(d_edges, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("Constructing R edges: " + str(len(r_list) * n_rand_neighbours))
        r_edges = []
        r_edges_created = 0
        for u in r_list:
            rand_neighbours = random.sample(r_list, n_rand_neighbours)
            while u in rand_neighbours:
                rand_neighbours = random.sample(r_list, n_rand_neighbours)

            for v in rand_neighbours:
                r_edges.append((u, v))
                r_edges.append((v, u))
            r_edges_created += n_rand_neighbours

            if r_edges_created % print_every_n == 0:
                print(r_edges_created)        
        
        with open('r_edge_list.pkl', 'wb') as handle:
            pickle.dump(r_edges, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_args(args):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--follower-files', dest='follower_files', nargs='+', default=[], help='list of csv follower files')
    parser.add_argument('--tweet-index-dict-file', dest='tweet_index_dict_file', type=str, default='', help='tweet to index dict pickle file')
    parser.add_argument('--tweet-table-csv', dest='tweet_table_csv', type=str, default='', help='tweet table csv file')
    parser.add_argument('--user-tweet-dict-file', dest='user_tweet_dict_file', type=str, default='', help='user to tweetids dict pickle file')
    parser.add_argument('--d-list', dest='d_list', type=str, default='', help='list of d follower tweets')
    parser.add_argument('--r-list', dest='r_list', type=str, default='', help='list of r follower tweets')
    return parser.parse_args(args)

if __name__ == '__main__':
    args = read_args(sys.argv[1:])

    follower_dicts = FollowerDictsEngine(args)

    #follower_dicts.create_userid_to_tweetids_dict()
    #follower_dicts.create_dr_tweet_list()
    follower_dicts.create_edge_list()
    

