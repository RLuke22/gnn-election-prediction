import re 
import pickle 
import numpy as np 
import pandas as pd 
import time

import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

use_gpus = torch.cuda.is_available()
if use_gpus:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

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

        # taken from https://www.kaggle.com/stoicstatic/twitter-sentiment-analysis-for-beginners
        self.url_pattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
        self.user_pattern = '@[^\s]+'
        self.alpha_pattern = "[^a-zA-Z0-9]"
        self.sequence_pattern = r"(.)\1\1+"
        self.seq_replace_pattern = r"\1\1"
        self.lemmatizer = WordNetLemmatizer()
        
    def clean(self, text_list):
        cleaned_text = []

        print("Cleaning data...")
        for tweet in tqdm(text_list):
            
            # convert to lowercase
            tweet = tweet.lower()

            # replace URLs with the word ' URL'
            tweet = re.sub(self.url_pattern, ' URL', tweet)
            # replace emojis 
            for emoji in self.emojis.keys():
                tweet = tweet.replace(emoji, "EMOJI" + self.emojis[emoji])
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

    def preprocess_election2020(self):
        model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

        sentences = ['#This framework! generates embeddings for each input sentence.',
                        'Sentences are passed as a list of string.', 
                        'The quick brown fox jumps over the lazy dog.']
        print("Generating embeddings...")
        start = time.time()
        sentence_embeddings = model.encode(sentences)
        end = time.time()
        print(end - start)

        print(sentence_embeddings.shape)



