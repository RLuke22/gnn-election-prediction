import pymongo

client = pymongo.MongoClient("mongodb+srv://Quintillion:TzjTGcE5I6Bu7P9e@twitterdata.wkwqp.mongodb.net/TwitterData?retryWrites=true&w=majority")
db = client["TwitterData"]
collection = db["Tweets"]

row = {"tweetid":1, "userid":1, "content":"quinton", "time":"8:00", "geotag":"somewhere", "party":"R", "sentiment":"N"}

x = collection.insert_one(row)
print(x.inserted_id)