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

hashtags_d_leaning = ['#biden2020', '#bidenharris2020', '#votebidenharris2020', '#theresistance', '#resist', '#votebluetosaveamerica',
                        '#votebiden', '#voteblue', '#bidenharrislandslide2020', '#votehimout', '#dumptrump']
hashtags_r_leaning = ['#trump2020', '#trumppence2020', '#maga', '#maga2020', '#kag', '#voteredtosaveamerica', '#votetrump', 
                        '#votered', '#trump2020landslide', '#sleepyjoe', '#neverbiden']
hashtags_n_leaning = ['#biden', '#bidenharris', '#joebiden', '#kamalaharris', '#trump', '#trumppence', '#donaldtrump', '#mikepence']

hashtags = hashtags_dem + hashtags_rep
query = ' OR '.join(hashtags)

geocodes = {'Arizona': '33.8244,-111.5818,146.4mi', 
            'Florida1':'27.1984,-83.0723,251.73mi',
            'Florida2':'29.4065,-86.1746,114mi',
            'Iowa1': '42.7802,-95.5281,51.26mi', 
            'Iowa2': '42.0493,-92.8059,100.05mi', 
            'Iowa3': '41.87931,-90.7568,27.67mi', 
            'Iowa4': '41.2540,-95.1875,37.76mi', 
            'Georgia1': '33.5816,-83.9150,78.99mi', 
            'Georgia2': '32.1667,-82.7974,109.92mi',
            'Alaska1': '64.3631,-155.8710,440.05mi',
            'Alaska2': '57.5656,-136.5903,132.45mi',
            'Alaska3': '55.5065,-132.1958,76.03mi',
            'Ohio1': '39.1930,-84.6620,8.81mi',
            'Ohio2': '39.8454,-83.3036,79.47mi',
            'Ohio3': '41.2866,-83.3382,34.05mi',
            'Ohio4': '41.3320,-81.6402,56.99mi',
            'Texas1': '27.7193,-97.6058.50mi,124.43mi',
            'Texas2': '30.8456,-96.8243 ,187.81mi',
            'Texas3': '31.8902,-106.4866,8.45mi',
            'Texas4': '32.4299,-100.4680,147.90mi',
            'NorthCarolina1': '35.3129,-78.2647,84.16mi',
            'NorthCarolina2': '35.6695,-80.1571,58.10mi',
            'NorthCarolina3': '35.7695,-81.5075,38.31mi',
            'NorthCarolina4': '35.5960,-82.4469,26.29mi'}

tweet_count = 5
for tweet_info in tweepy.Cursor(api.search, q=query, count=tweet_count, result_type="recent", tweet_mode="extended", geocode=geocodes['Alaska1'], lang='en').items(tweet_count):
    if 'retweeted_status' in dir(tweet_info):
        tweet=tweet_info.retweeted_status.full_text
    else:
        tweet=tweet_info.full_text

    print("--------------------------------------------------------------------")
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

                                                            

