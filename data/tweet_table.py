import pymongo
import tweepy
import inspect

client = pymongo.MongoClient("mongodb+srv://Quintillion:TzjTGcE5I6Bu7P9e@twitterdata.wkwqp.mongodb.net/TwitterData?retryWrites=true&w=majority")
db = client["TwitterData"]
collection = db["Tweets"]

""" row = {"tweetid":1, "userid":1, "text":"quinton", "created_at":"8:00", "geotag":"somewhere", "party":"R", "sentiment":"N"}

x = collection.insert_one(row)
print(x.inserted_id) """

consumer_key_Q = "DT3XJxLbG9gSAExLTnw8Ya8wy"
consumer_secret_Q = "9zhg6sRhqaBBgzYyBuoaohPH6WLpYEMWkg3S0koAhr50dfZQ1q"

auth = tweepy.AppAuthHandler(consumer_key_Q, consumer_secret_Q)
api = tweepy.API(auth)

query = '#biden OR #biden2020 OR #trump OR #trump2020'

# useful members in tweet object: created_at, id, text, 

for tweet in tweepy.Cursor(api.search, q=query, count = 1, result_type = "mixed").items(1):
    #print(inspect.getmembers(tweet))
    print(tweet.text)
    print(tweet.id)
    print(tweet.user.id)
    print(tweet.created_at)
    print([hashtag['text'] for hashtag in tweet.entities['hashtags']]) # this is a list of all the hashtags in the tweet. 
                                                                       # We can get party and sentiment by creating a dict of hashtags to "(D or R), (N, +, -)"


