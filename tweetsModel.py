# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 01:26:59 2016

@author: simon_hua
"""

from __future__ import division
import pandas as pd
import nltk
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder

stops = set(stopwords.words('english'))

def string2tokens(string):
    # regular expressions
    string = string.lower()
    string = re.sub(r"\@[a-z]+", "", string)  # remove the @jetblue...
    string = re.sub(r"http\S+", "", string)  # remove http....
    string = re.sub(r"&lt;3", "heartemojii", string)  # heartemojii
    string = re.sub(r"&amp", "&", string)
    string = re.sub(r"[^a-z0-9\s]", "", string)  # remove most punctuation

    string = string.decode('utf8')
    tokens = nltk.word_tokenize(string)
    return (tokens)

def extract_onewords(tweets, min_support = 1):
    #extract the texts into a list of tweets
    list_tweets = tweets.text.tolist()
    final_text = ""

    #append each tweet to make a giant string
    for i in range(len(list_tweets)):
        list_tweets[i] = list_tweets[i].lower()
        final_text = final_text + " " + list_tweets[i]

    tokens = string2tokens(final_text)

    #get rid of stopwords
    dummy1 = [] 
    for token in tokens:
      if token not in stops:
          dummy1.append(token)
  
    tokens = dummy1
    
    final_nltk = nltk.Text(tokens)

    #Frequency distribution of the most common words
    fdist = nltk.FreqDist(final_nltk)
    #print fdist.most_common(100)
    
    #return a dictionary of only the words with support more than min support
    dic = { k:v for k, v in fdist.items() if v >= min_support }

    return Counter(dic)

def extract_bigrams(tweets, min_support = 1):
    #extract the texts into a list of tweets
    list_tweets = tweets.text.tolist()
    final_text = ""

    #append each tweet to make a giant string
    for i in range(len(list_tweets)):
        list_tweets[i] = list_tweets[i].lower()
        final_text = final_text + " " + list_tweets[i]

    tokens = string2tokens(final_text)
    final_nltk = nltk.Text(tokens)

    finder = BigramCollocationFinder.from_words(final_nltk)

    #only bigrams that satisfy minimum support
    finder.apply_freq_filter(min_support)

    finder = finder.ngram_fd.items()

    bigrams = {z[0]:z[1] for z in finder}

    return Counter(bigrams)

def extract_trigrams(tweets, min_support = 1):
    #extract the texts into a list of tweets
    list_tweets = tweets.text.tolist()
    final_text = ""

    #append each tweet to make a giant string
    for i in range(len(list_tweets)):
        list_tweets[i] = list_tweets[i].lower()
        final_text = final_text + " " + list_tweets[i]

    tokens = string2tokens(final_text)

    final_nltk = nltk.Text(tokens)

    finder = TrigramCollocationFinder.from_words(final_nltk)

    #only bigrams that satisfy minimum support
    finder.apply_freq_filter(min_support)

    finder = finder.ngram_fd.items()

    trigrams = {z[0]:z[1] for z in finder}

    return Counter(trigrams)


################################################################################
# Scoring System
################################################################################

class TweetScoreMachine:
    def __init__(self, fd, bg = None, tg = None):
        self.freq_dist = fd
        self.bigram_dist = bg
        self.trigram_dist = tg
        self.sents = ['pos', 'neu', 'neg']

        self.n_freq_dist = Counter()
        self.n_bigram_dist = Counter()
        self.n_trigram_dist = Counter()

        for sent in self.sents:
            # for key, count in fd[sent].iteritems():
            #     self.n_freq_dist[sent] += count
            # for key, count in bg[sent].iteritems():
            #     self.n_bigram_dist[sent] += count
            # for key, count in tg[sent].iteritems():
            #     self.n_trigram_dist[sent] += count
            self.n_freq_dist[sent] = max(fd[sent].values())
            self.n_bigram_dist[sent] = max(bg[sent].values())
            self.n_trigram_dist[sent] = max(tg[sent].values())

    # Get the score of a list of tokens. Scoring is based on the frequency of a negative in each of the bags
    # (i.e. if a word appears 172 times in a bag, then that word will count for 172 points.
    def get_score(self, tweet_tokens, bigrams = 0, trigrams = 0, bigram_factor = 2, trigram_factor = 3):
        scores = {}
        for sent in self.sents:
            scores[sent] = 0.0

        # For each word in the tweet, check to see if it is in the freq. dist.; if it is,
        # add the normalized points of it to the total score.
        for token in tweet_tokens:
            for sent in self.sents:
                scores[sent] += self.freq_dist[sent][token] / self.n_freq_dist[sent]

        # Factor in bigrams if the option is set.
        if (bigrams):
            for sent in self.sents:
                for i in range(0, len(tweet_tokens) - 1):
                    bigram = (tweet_tokens[i], tweet_tokens[i+1])
                    scores[sent] += self.bigram_dist[sent][bigram] / self.n_bigram_dist[sent] * bigram_factor

        # Factor in trigrams if the option is set.
        if (trigrams):
            for sent in self.sents:
                for i in range(0, len(tweet_tokens) - 2):
                    trigram = (tweet_tokens[i], tweet_tokens[i+1], tweet_tokens[i+2])
                    scores[sent] += self.trigram_dist[sent][trigram] / self.n_trigram_dist[sent] * trigram_factor

        # Normalize the scores.
        total_score = 0.0
        for sent in self.sents:
            total_score += scores[sent]
        for sent in self.sents:
            if (total_score != 0.0):
                scores[sent] /= total_score

        # Return as a tuple.
        return scores

def find_sentiment(score):
    max_val = 0.0
    max_sent = ""
    for k, v in score.iteritems():
        if (max_val < v):
            max_val = v
            max_sent = k

    # if (abs(score['pos'] - score['neg']) < 0.1 and score['pos'] > 0.3 and score['neg'] > 0.3):
    #     return "neutral"
    if (score['pos'] == 0.0 and score['neu'] == 0.0 and score['neg'] == 0.0):
        return "neutral"
    # if (score['pos'] < 0.35 and score['neu'] < 0.35 and score['neg'] < 0.35):
    #     return "neutral"

    if (max_sent == "pos"):
        return "positive"
    elif (max_sent == "neu"):
        return "neutral"
    else:
        return "negative"

################################################################################
# Procedural Code
################################################################################

file_dir_to_read =""
file_to_read = "tweets_stocks_testing"

# Read in the tweets from Kaggle.
tweets = pd.read_csv("Kaggle/Tweets.csv")
# tweets_df = pd.read_csv(file_dir_to_read + file_to_read + ".csv")
tweets_df = tweets

# Global parameters to play with...
min_supp_onewords = 1
min_supp_bigrams = 1
min_supp_trigrams = 1
use_bigrams = 1
use_trigrams = 1
bigram_factor = 2
trigram_factor = 3

#separate into positive, neutral, and negative tweets
pos_tweets = tweets[tweets.airline_sentiment == "positive"]
neu_tweets = tweets[tweets.airline_sentiment == "neutral"]
neg_tweets = tweets[tweets.airline_sentiment == "negative"]

oneword_pos = extract_onewords(pos_tweets, min_supp_onewords)
oneword_neu = extract_onewords(neu_tweets, min_supp_onewords)
oneword_neg = extract_onewords(neg_tweets, min_supp_onewords)

bigrams_pos = extract_bigrams(pos_tweets, min_supp_bigrams)
bigrams_neu = extract_bigrams(neu_tweets, min_supp_bigrams)
bigrams_neg = extract_bigrams(neg_tweets, min_supp_bigrams)

trigrams_pos = extract_trigrams(pos_tweets, min_supp_trigrams)
trigrams_neu = extract_trigrams(neu_tweets, min_supp_trigrams)
trigrams_neg = extract_trigrams(neg_tweets, min_supp_trigrams)

fd = {'pos': oneword_pos, 'neu': oneword_neu, 'neg': oneword_neg}
bg = {'pos': bigrams_pos, 'neu': bigrams_neu, 'neg': bigrams_neg}
tg = {'pos': trigrams_pos, 'neu': trigrams_neu, 'neg': trigrams_neg}

score_machine = TweetScoreMachine(fd, bg, tg)

# f = open("file.txt", "w")
count_match = 0
n_tweets = len(tweets_df)
for i in range(0, n_tweets):
    tweet_text = tweets_df.loc[i, 'text']
    score = score_machine.get_score(string2tokens(tweet_text),
                                    use_bigrams,
                                    use_trigrams,
                                    bigram_factor,
                                    trigram_factor
                                    )
    # tweet_line_print = "(should be: " + tweet_sentiment + ") | (is actually: " + find_sentiment(score) + ") " + str(score)
    # tweet_line_print = "(" + find_sentiment(score) + ") " + str(score) + " | \"" + tweet_text + "\""
    # print tweet_line_print
    tweet_sentiment = tweets_df.loc[i, 'airline_sentiment']
    tweets_df.loc[i, 'Actual.Sentiment'] = tweet_sentiment
    tweets_df.loc[i, 'Predicted.Sentiment'] = find_sentiment(score)
    tweets_df.loc[i, 'Score.Positive'] = score['pos']
    tweets_df.loc[i, 'Score.Neutral'] = score['neu']
    tweets_df.loc[i, 'Score.Negative'] = score['neg']

    # print >> f, tweet_line_print

    # Count the matching sentiments...
    if (find_sentiment(score) == tweet_sentiment):
        count_match += 1

print "Min. Support: " + str(1) + "; Accuracy: " + str(count_match / n_tweets * 100.0) + "%"
# f.close()

# tweets_df.to_csv('tweets_stocks_testing.csv', index=False, sep=',')
# tweets_df.to_csv("Output/" + file_to_read + "_new.csv", index=False, sep=',')
