# We use this file to perform manual annotation on approximately 20000 tweets

import argparse
import sys 
import os 
import re

import pymongo

class DataAnnotationEngine():
    def __init__(self, args):
        super(DataAnnotationEngine, self).__init__()

        self.n_annotating = args.n_annotating 
        self.client = pymongo.MongoClient("mongodb+srv://Quintillion:TzjTGcE5I6Bu7P9e@twitterdata.wkwqp.mongodb.net/TwitterData?retryWrites=true&w=majority")
        self.db = self.client['TwitterData']
        self.collection = self.db['Tweets']

        # We only retrieve entries labelled UU for manual annotation
        self.retrieve_label = 'UU'

        # We separate manually annotated labels from automatically annotated labels
        # in case we want to separate downstream analysis
        self.neutral_ann = 'N'
        self.republican_ann = 'RR'
        self.democrat_ann = 'DD'

    def annotate_data(self):

        for i in range(self.n_annotating):
            if not (i+1) % 500 == 0:
                print("Annotation counter: {}/{}".format(i+1, self.n_annotating))
            else:
                # Some positive reinforcement to prevent us from giving up
                # during the manual annotation process
                print("Wow! You've annotated {} tweets! You're a rockstar!".format(i+1))

            cursor = self.collection.aggregate(
                [
                    { '$match': { 'party': 'UU' } }, 
                    { '$sample': { 'size': 1 } }
                ]
            )

            for sample in cursor:
                print("------------------------------------------------------------")
                print(sample['tweet_id'])
                print()
                print(sample['text'])
                print("------------------------------------------------------------\n")

                response_valid = False
                while not response_valid:
                    response = input("Party (d/r/n): ")

                    if response == 'd':
                        party_label = self.democrat_ann
                        response_valid = True
                    elif response == 'r':
                        party_label = self.republican_ann
                        response_valid = True
                    elif response == 'n':
                        party_label = self.neutral_ann 
                        response_valid = True 

                self.collection.update_one(
                    { '_id': sample['_id'] }, 
                    { '$set': {'party': party_label}}
                )

def read_args(args):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-annotating', dest='n_annotating', type=int, default=5000)
    
    return parser.parse_args(args)

if __name__ == '__main__':
    args = read_args(sys.argv[1:])

    print("Polling configurations")
    print("_____________________")
    print("Num Annotating: ", args.n_annotating)
    print()

    data_engine = DataAnnotationEngine(args)

    # We annotate data in the command line
    data_engine.annotate_data()




