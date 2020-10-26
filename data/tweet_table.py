import pymongo
import tweepy

client = pymongo.MongoClient("mongodb+srv://Quintillion:TzjTGcE5I6Bu7P9e@twitterdata.wkwqp.mongodb.net/TwitterData?retryWrites=true&w=majority")
db = client["TwitterData"]
collection = db["Tweets"]

""" row = {"tweetid":1, "userid":1, "content":"quinton", "time":"8:00", "geotag":"somewhere", "party":"R", "sentiment":"N"}

x = collection.insert_one(row)
print(x.inserted_id) """

consumer_key_Q = "DT3XJxLbG9gSAExLTnw8Ya8wy"
consumer_secret_Q = "9zhg6sRhqaBBgzYyBuoaohPH6WLpYEMWkg3S0koAhr50dfZQ1q"

auth = tweepy.AppAuthHandler(consumer_key_Q, consumer_secret_Q)
api = tweepy.API(auth)

query = '#biden OR #biden2020 OR #trump OR #trump2020'

for tweet in tweepy.Cursor(api.search, q=query, count = 1, result_type = "mixed").items(1):
    print(tweet.text)