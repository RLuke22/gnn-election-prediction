import time
import tweepy

consumer_key_Q = "JZNzXeOD8VMDCroiKFXsXwAdg"
consumer_secret_Q = "O58HHCidMQ8bogw4ofs8hr50V45aYxAG2i9vvqfZBFPXI3zKjM"
api = tweepy.API(tweepy.AppAuthHandler(consumer_key_Q, consumer_secret_Q))

ids = []
for page in tweepy.Cursor(api.followers_ids, screen_name="realDonaldTrump").pages():
    ids.extend(page)
    break 

print(ids)
print(len(ids))    