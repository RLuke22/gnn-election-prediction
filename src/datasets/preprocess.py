import re 
import pickle 
import emoji
# nltk.download('wordnet') # run only first time
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

class DataPreprocessor():
    def __init__(self):
        # taken from https://www.kaggle.com/stoicstatic/twitter-sentiment-analysis-for-beginners
        self.emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

        # taken from https://www.kaggle.com/stoicstatic/twitter-sentiment-analysis-for-beginners
        self.stopword_list = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
                    'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
                    'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
                    'does', 'doing', 'down', 'during', 'each','few', 'for', 'from', 
                    'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
                    'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
                    'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
                    'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
                    'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're',
                    's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
                    't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
                    'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 
                    'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
                    'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
                    'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
                    "youve", 'your', 'yours', 'yourself', 'yourselves']

        self.hashtag_list = [
            '#votebluetosaveamerica',
            '#voteredtosaveamerica2020',
            '#voteredtosaveamerica',
            '#joebiden2020',
            '#votehimout2020',
            '#trump2020nowmorethanever',
            '#sleepyjoe', 
            '#neverbiden',
            '#votetrumppence',
            '#fourmoreyears',
            '#votered2020',
            '#trump2020landslide',
            '#magaa',
            '#keepamericagreat',
            '#donaldtrump2020',
            '#makeamericagreatagain',
            '#kag2020',
            '#trumptrain',
            '#trump2020tosaveamerica',
            '#trump2020', 
            '#trumppence2020',
            '#hunterbiden',
            '#teamtrump',
            '#americafirst',
            '#votebidenharris2020', 
            '#theresistance', 
            '#resist', 
            '#votehimout', 
            '#dumptrump',
            '#votebidenharris',
            '#votebluetoendthisnightmare',
            '#voteblue2020',
            '#bidenharrislandslide2020',
            '#buildbackbetter',
            '#votebidenharristosaveamerica', 
            '#votebluenomatterwho',
            '#bidenharris2020landslide',
            '#bluewave2020',
            '#bidenharristosaveamerica',
            '#biden2020', 
            '#bidenharris2020',
            '#votered', 
            '#teambiden',
            '#trumppence',
            '#bidenharris',
            '#votetrump',
            '#voteblue', 
            '#maga2020', 
            '#votebiden',
            '#bluewave',
            '#maga', 
            '#kag', 
        ]

        # taken from https://www.kaggle.com/stoicstatic/twitter-sentiment-analysis-for-beginners
        self.url_pattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
        self.user_pattern = '@[^\s]+'
        self.alpha_pattern = "[^a-zA-Z0-9]"
        self.sequence_pattern = r"(.)\1\1+"
        self.seq_replace_pattern = r"\1\1"
        self.lemmatizer = WordNetLemmatizer()

    def twitter_clean(self, text_list):
        cleaned_text = []

        print("Cleaning Twitter data...")
        for tweet in tqdm(text_list):

            # remove all "ground-truthy" hashtags
            # We do not want the network to learn the hashtag->party mapping
            # as several tweets in our dataset are unlabelled and do not have hashtags
            for hashtag in self.hashtag_list:
                tweet = re.sub(r'%s\b' % re.escape(hashtag), 'HASHTAG', tweet.lower())
            
            # All but the following hashtags are removed
            # For these hashtags, they are replaced with their corresponding English words
            tweet = re.sub(r'#biden\b', 'biden', tweet)
            tweet = re.sub(r'#trump\b', 'trump', tweet)
            tweet = re.sub(r'#joebiden\b', 'joe biden', tweet)
            tweet = re.sub(r'#kamalaharris\b', 'kamala harris', tweet)
            tweet = re.sub(r'#donaldtrump\b', 'donald trump', tweet)
            tweet = re.sub(r'#mikepence\b', 'mike pence', tweet)
            tweet = re.sub(r'#democrats\b', 'democrats', tweet)
            tweet = re.sub(r'#democrat\b', 'democrat', tweet)
            tweet = re.sub(r'#republicans\b', 'republicans', tweet)
            tweet = re.sub(r'#republican\b', 'republican', tweet)

            cleaned_text.append(tweet)
        
        return cleaned_text
        
    def clean(self, text_list):
        cleaned_text = []

        print("Cleaning data...")
        for tweet in tqdm(text_list):
            # convert to lowercase
            tweet = tweet.lower()

            # first remove emojis
            tweet = emoji.demojize(tweet, delimiters=("", ""))

            # replace URLs with the word ' URL'
            tweet = re.sub(self.url_pattern, ' URL', tweet)
            # replace emojis 
            for emoticon in self.emojis.keys():
                tweet = tweet.replace(emoticon, "EMOJI" + self.emojis[emoticon])
            # replace @USERNAME with ' USER'
            tweet = re.sub(self.user_pattern, ' USER', tweet)
            # replace all non-alphabets
            tweet = re.sub(self.alpha_pattern, ' ', tweet)
            # replace 3 or more consecutive letters by 2 letters
            tweet = re.sub(self.sequence_pattern, self.seq_replace_pattern, tweet)

            # lemmatize words and remove stopwords
            tweet_words = ''
            for word in tweet.split():
                if word not in self.stopword_list and len(word) > 1:
                    word = self.lemmatizer.lemmatize(word)
                    tweet_words += (word + ' ')

            cleaned_text.append(tweet_words)

        return cleaned_text



