import argparse
import sys
import os
import pymongo
import csv

client = pymongo.MongoClient("mongodb+srv://Quintillion:TzjTGcE5I6Bu7P9e@twitterdata.wkwqp.mongodb.net/TwitterData?retryWrites=true&w=majority")
db = client['TwitterData']
collection = db['Tweets']

toy_dataset = "./toy_dataset.csv"
written = 0
write_every_n = 5000

print("Writing to toy dataset.")

for document in collection.find():
    text = document['text']
    party = document['party']

    if party != "U" and party != "UU":
        with open(toy_dataset, 'a+') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['text', 'party'])
            row = {'text':text, 'party':party}
            writer.writerow(row)

        written += 1
        if written % write_every_n == 0:
            print("Written " + str(write_every_n) + " to toy dataset.")

