import pymongo
import tweepy
import inspect
from tqdm import tqdm
import numpy as np

# client = pymongo.MongoClient("mongodb+srv://Quintillion:TzjTGcE5I6Bu7P9e@twitterdata.wkwqp.mongodb.net/TwitterData?retryWrites=true&w=majority")
# db = client["TwitterData"]
# collection = db["Tweets"]

# row = {"tweetid":1, "userid":1, "text":"quinton quinton quinton quinton quinton quinton quinton quinton quinton quinton quinton quinton quinton quinton quinton quinton", "created_at":"8:00", "state":"AZ", "party":"R", "user_processed":1}

# for i in tqdm(range(100000)):
#     if i % 3 == 0:
#         user_id = 42 
#     else:
#         user_id = i
#     row = {"tweetid":i, "userid":user_id, "text":"quinton quinton quinton quinton quinton quinton quinton quinton quinton quinton quinton quinton quinton quinton quinton quinton", "created_at":"8:00", "state":"AZ", "party":"R", "user_processed":1}
#     x = collection.insert_one(row)

# exit(0)

# Luke's keys
consumer_key_Q = "JZNzXeOD8VMDCroiKFXsXwAdg"
consumer_secret_Q = "O58HHCidMQ8bogw4ofs8hr50V45aYxAG2i9vvqfZBFPXI3zKjM"

auth = tweepy.AppAuthHandler(consumer_key_Q, consumer_secret_Q)
api = tweepy.API(auth)

hashtags_d_leaning = [ 
    '#votebidenharris2020', 
    '#theresistance', 
    '#resist', 
    '#votebiden', 
    '#voteblue', 
    '#votehimout', 
    '#dumptrump'
]
hashtags_r_leaning = [ 
    '#maga', 
    '#maga2020', 
    '#kag', 
    '#votetrump', 
    '#votered', 
    '#sleepyjoe', 
    '#neverbiden'
]
hashtags_n_leaning = [
    '#trump2020', 
    '#trumppence2020',
    '#biden2020', 
    '#bidenharris2020',
    '#biden', 
    '#bidenharris', 
    '#joebiden', 
    '#kamalaharris', 
    '#trump', 
    '#trumppence', 
    '#donaldtrump', 
    '#mikepence'
]

hashtags_d_leaning = [
    '#votebluetosaveamerica',
    '#votebidenharris',
    '#votebluetoendthisnightmare',
    '#voteblue2020',
    '#bidenharrislandslide2020',
    '#buildbackbetter',
    '#votebidenharristosaveamerica',
]
hashtags_r_leaning = [
    '#voteredtosaveamerica',
    '#votetrumppence',
    '#fourmoreyears',
    '#votered2020',
    '#trump2020landslide',
    '#magaa',
    '#keepamericagreat'
]
hashtags_n_leaning = [
    '#democrats',
    '#democrat',
    '#republicans',
    '#republican'
]

hashtags = hashtags_d_leaning + hashtags_r_leaning + hashtags_n_leaning
query = ' OR '.join(hashtags)

geocodes = {'Arizona0': '33.8244,-111.5818,146.4mi', 
            'Florida0':'27.1984,-83.0723,251.73mi',
            'Florida1':'29.4065,-86.1746,114mi',
            'Iowa0': '42.7802,-95.5281,51.26mi', 
            'Iowa1': '42.0493,-92.8059,100.05mi', 
            'Iowa2': '41.87931,-90.7568,27.67mi', 
            'Iowa3': '41.2540,-95.1875,37.76mi', 
            'Georgia0': '33.5816,-83.9150,78.99mi', 
            'Georgia1': '32.1667,-82.7974,109.92mi',
            'Ohio0': '39.1930,-84.6620,8.81mi',
            'Ohio1': '39.8454,-83.3036,79.47mi',
            'Ohio2': '41.2866,-83.3382,34.05mi',
            'Ohio3': '41.3320,-81.6402,56.99mi',
            'Texas0': '27.7193,-97.6058.50mi,124.43mi',
            'Texas1': '30.8456,-96.8243 ,187.81mi',
            'Texas2': '31.8902,-106.4866,8.45mi',
            'Texas3': '32.4299,-100.4680,147.90mi',
            'NorthCarolina0': '35.3129,-78.2647,84.16mi',
            'NorthCarolina1': '35.6695,-80.1571,58.10mi',
            'NorthCarolina2': '35.7695,-81.5075,38.31mi',
            'NorthCarolina3': '35.5960,-82.4469,26.29mi'}

# #These arrays define a multinomial distribution defining the relative populations of each region (per state)
# state_region_probs = {'Arizona': [1],
#                     'Florida': [0.9487, 0.0513],
#                     'Iowa': [0.1157, 0.7226, 0.0982, 0.06305],
#                     'Georgia': [0.7444, 0.2556],
#                     'Ohio': [0.1073, 0.4515, 0.0703, 0.3709],
#                     'Texas': [0.1290, 0.7860, 0.0284, 0.0566],
#                     'NorthCarolina': [0.4508, 0.3790, 0.1170, 0.0532]}

tweet_count = 200
for i, tweet_info in enumerate(tweepy.Cursor(api.search, 
                                q=query, 
                                count=tweet_count, 
                                result_type="recent", 
                                tweet_mode="extended", 
                                geocode=geocodes['Ohio1'], 
                                lang='en',
                                until='2020-10-21'
                                ).items(tweet_count)):
    
    if 'retweeted_status' in dir(tweet_info):
        tweet=tweet_info.retweeted_status.full_text
    else:
        tweet=tweet_info.full_text

    print("--------------------------------------------------------------------")
    print(i)
    print(tweet)
    print(tweet_info.id)
    print(tweet_info.user.id)
    print(tweet_info.created_at)
    
    # for h in tweet.split() where h[0] is '#'
    #     for hashtag in h.split('#') // this is to deal with the #something#something cases
    #         add '#' + hashtag to list if hashtag is not empty
    tweet_hashtags =  [('#' + hashtag) for hsplit in [h.split('#') for h in tweet.split() if h[0] == '#'] for hashtag in hsplit if hashtag != '']
    print(tweet_hashtags)

    d_hashes = [h for h in tweet_hashtags if h.lower() in hashtags_d_leaning]
    r_hashes = [h for h in tweet_hashtags if h.lower() in hashtags_r_leaning]
    n_hashes = [h for h in tweet_hashtags if h.lower() in hashtags_n_leaning]

    if d_hashes and r_hashes:
        print('U')
    elif d_hashes:
        print('D')
    elif r_hashes:
        print('R')
    else:
        print('U')

    print()

                                                            

