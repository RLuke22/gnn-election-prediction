import pymongo
import tweepy
import inspect

client = pymongo.MongoClient("mongodb+srv://Quintillion:TzjTGcE5I6Bu7P9e@twitterdata.wkwqp.mongodb.net/TwitterData?retryWrites=true&w=majority")
db = client["TwitterData"]
collection = db["Tweets"]

""" row = {"tweetid":1, "userid":1, "text":"quinton", "created_at":"8:00", "geotag":"somewhere", "party":"R", "sentiment":"N"}

x = collection.insert_one(row)
print(x.inserted_id) """

# Luke's keys
consumer_key_Q = "JZNzXeOD8VMDCroiKFXsXwAdg"
consumer_secret_Q = "O58HHCidMQ8bogw4ofs8hr50V45aYxAG2i9vvqfZBFPXI3zKjM"

auth = tweepy.AppAuthHandler(consumer_key_Q, consumer_secret_Q)
api = tweepy.API(auth)

hashtags_dem = ['#biden', 
                '#biden2020', 
                '#bidenharris', 
                '#bidenharris2020', 
                '#joebiden', 
                '#kamalaharris',
                '#votebidenharris2020',
                '#theresistance',
                '#resist',
                '#votebluetosaveamerica',
                '#votebiden',
                '#voteblue',
                '#bidenharrislandslide2020',
                '#sleepyjoe',
                '#neverbiden']

hashtags_rep = ['#trump',
                '#trump2020',
                '#trumppence',
                '#trumppence2020',
                '#donaldtrump',
                '#mikepence',
                '#maga',
                '#maga2020',
                '#kag',
                '#voteredtosaveamerica',
                '#votetrump',
                '#votered',
                '#trump2020landslide',
                '#votehimout',
                '#dumptrump']

hashtags = hashtags_dem + hashtags_rep
# currently query is 471 characters
query = ' OR '.join(hashtags)

tweet_count = 5
for tweet_info in tweepy.Cursor(api.search, q=query, count = tweet_count, result_type = "recent", tweet_mode="extended").items(tweet_count):
    if 'retweeted_status' in dir(tweet_info):
        tweet=tweet_info.retweeted_status.full_text
    else:
        tweet=tweet_info.full_text

    print("--------------------------------------------------------------------")
    print(tweet)
    print(tweet_info.id)
    print(tweet_info.user.id)
    print(tweet_info.created_at)
    # We can get party and sentiment by creating a dict of hashtags to "(D or R), (N, +, -)"
    print([hashtag['text'] for hashtag in tweet_info.entities['hashtags']])
    print()

                                                            

