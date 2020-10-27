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
                '#votetrump',
                '#votered',
                '#trump2020landslide',
                '#votehimout',
                '#dumptrump']

hashtags = hashtags_dem + hashtags_rep
query = ' OR '.join(hashtags)

geocodes = {'Arizona': '33.8244,-111.5818,146.4mi', 
            'Iowa1': '42.7802,-95.5281,51.26mi', 
            'Iowa2': '42.0493,-92.8059,100.05mi', 
            'Iowa3': '41.87931,-90.7568,27.67mi', 
            'Iowa4': '41.2540,-95.1875,37.76mi', 
            'Georgia': '33.5816,-83.9150,78.99mi', 
            'Alaska1': '64.36,-155.87,440.05mi',
            'Alaska2': '57.57,-136.59,132.45mi',
            'Alaska3': '55.506,-132.196,76.03mi',
            'Ohio1': '39.1973,-84.5960,7.93mi',
            'Ohio2': '39.8454,-83.3036,77.99mi',
            'Ohio3': '41.3320,-83.1130,31.78mi',
            'Ohio4': '41.3320,-81.6402,56.99mi',
            'Texas1': '27.7193,-97.6058,113.50mi',
            'Texas2': '30.7714,-96.7811,180.20mi',
            'Texas3': '32.4893,-95.5963,89.95mi',
            'Texas4': '31.8902,-106.4866,8.45mi',
            'NorthCarolina1': '35.3129,-78.2647,84.16mi',
            'NorthCarolina2': '35.6963,-80.1351,55.63mi',
            'NorthCarolina3': '35.7695,-81.5075,38.31mi',
            'NorthCarolina4': '35.5960,-82.4469,26.29mi'}

tweet_count = 5
for tweet_info in tweepy.Cursor(api.search, q=query, count = tweet_count, result_type = "recent", tweet_mode="extended", geocode=geocodes['Alaska1']).items(tweet_count):
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
    #print([hashtag['text'] for hashtag in tweet_info.entities['hashtags']])
    print()

                                                            

