import argparse
import sys
import os
import re
import csv

import pymongo
import tweepy

client = pymongo.MongoClient("mongodb+srv://Quintillion:TzjTGcE5I6Bu7P9e@twitterdata.wkwqp.mongodb.net/TwitterData?retryWrites=true&w=majority")
db = client['TwitterData']
collection = db['Tweets']

consumer_key_Q = "DT3XJxLbG9gSAExLTnw8Ya8wy"
consumer_secret_Q = "9zhg6sRhqaBBgzYyBuoaohPH6WLpYEMWkg3S0koAhr50dfZQ1q"
api = tweepy.API(tweepy.AppAuthHandler(consumer_key_Q, consumer_secret_Q), wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

hashtags_d_leaning = [ 
                '#votebidenharris2020', 
                '#theresistance', 
                '#resist', 
                '#votebiden', 
                '#voteblue', 
                '#votehimout', 
                '#dumptrump',
                '#votebluetosaveamerica',
                '#votebidenharris',
                '#votebluetoendthisnightmare',
                '#voteblue2020',
                '#bidenharrislandslide2020',
                '#buildbackbetter',
                '#votebidenharristosaveamerica',
                '#votebluenomatterwho',
                '#bidenharris2020landslide',
                '#bluewave',
                '#bluewave2020',
                '#bidenharristosaveamerica',
                '#joebiden2020',
                '#votehimout2020',
                '#biden2020', 
                '#bidenharris2020',
                '#teambiden',
            ]

hashtags_r_leaning = [ 
                '#maga', 
                '#maga2020', 
                '#kag', 
                '#votetrump', 
                '#votered', 
                '#sleepyjoe', 
                '#neverbiden',
                '#voteredtosaveamerica',
                '#votetrumppence',
                '#fourmoreyears',
                '#votered2020',
                '#trump2020landslide',
                '#magaa',
                '#keepamericagreat',
                '#trump2020nowmorethanever',
                '#donaldtrump2020',
                '#makeamericagreatagain',
                '#kag2020',
                '#trumptrain',
                '#voteredtosaveamerica2020',
                '#trump2020tosaveamerica',
                '#trump2020', 
                '#trumppence2020',
                '#hunterbiden',
                '#teamtrump',
                '#americafirst'
            ]

hashtags_n_leaning = [ 
                '#biden', 
                '#bidenharris', 
                '#joebiden', 
                '#kamalaharris', 
                '#trump', 
                '#trumppence', 
                '#donaldtrump', 
                '#mikepence',
                '#democrats',
                '#democrat',
                '#republicans',
                '#republican',
            ]

hashtags_uu = [
        'biden',
        'trump',
        'democrat',
        'republican',
        'democrats',
        'republicans'
]

drn_hashtags = hashtags_d_leaning + hashtags_r_leaning + hashtags_n_leaning

num_updated = 0
print_every_n = 500
csv_path = '../../election2020.csv'
fieldnames=[
    'tweet_id', 
    'user_id', 
    'retweet_user_id', 
    'text', 
    'party',
    'state', 
    'hashtags', 
    'keywords'
]

not_updated_documents = {'hashtags': {'$exists': False}}

with open(csv_path, 'a+') as csvfile:
    for document in collection.find(not_updated_documents):
        _id      = document['_id']
        tweet_id = document['tweet_id']
        user_id  = document['user_id']
        text     = document['text']
        state    = document['state']
        party    = document['party']
        retweet_user_id = document['retweet_user_id']
        
        hashtags = []
        for ht in drn_hashtags:
            if ht in text.lower():
                hashtags.append(ht)

        d_hashes = [h for h in hashtags if h.lower() in hashtags_d_leaning]
        r_hashes = [h for h in hashtags if h.lower() in hashtags_r_leaning]
        n_hashes = [h for h in hashtags if h.lower() in hashtags_n_leaning]
        if party != 'UU':
            if d_hashes and r_hashes:
                party = 'U'
            elif d_hashes:
                party = 'D'
            elif r_hashes:
                party = 'R'
            else:
                party = 'U'
        hashtags_string = ','.join(d_hashes + r_hashes + n_hashes)

        text_nohash = ' '.join([word for word in text.split() if not word.startswith('#')])
        uu_words = []
        for ht in hashtags_uu:
            if ht in text_nohash.lower():
                uu_words.append(ht)
        uu_words_string = ','.join(uu_words)

        document_search = { '_id': _id }
        update = { '$set': {'party': party, 'hashtags': hashtags_string, 'n_terms': uu_words_string} }

        collection.update_one(document_search, update)

        # Write to csv file
        if state in ['Arizona0', 'Arizona']:
            state = 'Arizona'
        elif state in ['Florida', 'Florida0', 'Florida1']:
            state = 'Florida'
        elif state in ['Iowa', 'Iowa0', 'Iowa1', 'Iowa2', 'Iowa3']:
            state = 'Iowa'
        elif state in ['Georgia', 'Georgia0', 'Georgia1']:
            state = 'Georgia'
        elif state in ['Ohio', 'Ohio0', 'Ohio1', 'Ohio2', 'Ohio3']:
            state = 'Ohio'
        elif state in ['Texas', 'Texas0', 'Texas1', 'Texas2', 'Texas3']:
            state = 'Texas'
        elif state in ['NorthCarolina', 'NorthCarolina0', 'NorthCarolina1', 'NorthCarolina2', 'NorthCarolina3']:
            state = 'NorthCarolina'
        elif state in ['Kansas', 'Kansas0', 'Kansas1']:
            state = 'Kansas'
        elif state in ['Oklahoma', 'Oklahoma0', 'Oklahoma1', 'Oklahoma2']:
            state = 'Oklahoma'
        elif state in ['Hawaii0', 'Hawaii']:
            state = 'Hawaii'
        else:
            print(state)
            break
        
        if party == 'UU' or party == 'N':
            party = 'U'
        elif party == 'RR':
            party = 'R'
        elif party == 'DD':
            party = 'D'

        hashtags = hashtags_string
        if len(hashtags) == 0:
            hashtags = 'none'
        n_terms  = uu_words_string
        if len(n_terms) == 0:
            n_terms = 'none'

        row = {}
        row['tweet_id'] = tweet_id 
        row['user_id'] = user_id
        row['retweet_user_id'] = retweet_user_id
        row['text'] = text 
        row['party'] = party
        row['state'] = state
        row['hashtags'] = hashtags
        row['keywords'] = n_terms
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(row)

        num_updated += 1
        if num_updated % print_every_n == 0:
            print("Updated " + str(num_updated) + " documents.")

        
        




        
        

