#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import nltk
import matplotlib.pyplot as plt
import math
import re
from nltk.corpus import stopwords as StopwordsLoader
from textblob import Word, TextBlob

import pickle
from tempfile import TemporaryFile

from nltk import bigrams

# reg for match url
url_reg = r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+'

# reg for match non-alphanumeric characters
alpha_reg = r'[^a-zA-Z\d\s]+'

# reg for match pure number words
num_reg = r'\b\d+(\.|\s+)?\d*\b'

# reg for match emotion
emoticons_reg = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

# reg for tokenize
regex_reg = [
    emoticons_reg,
    url_reg,                            # URLs
    r'<[^>]+>',                         # HTML tags
    r'(?:@[\w_]+)',                     # @-mentions
    r'(?:\#+[\w_]+[\w\'_\-]*[\w_]+)',   # hash-tags

    r"(?:[a-z][a-z'\-_]+[a-z])",        # words with - and '
    r'(?:[\w_]+)',                      # other words
    r'(?:\S)'                           # anything else
]

token_match = re.compile(r'('+'|'.join(regex_reg)+')',
                         re.VERBOSE | re.IGNORECASE)
emotion_match = re.compile(r'^'+emoticons_reg+'$', re.VERBOSE | re.IGNORECASE)
url_match = re.compile(url_reg, re.VERBOSE | re.IGNORECASE)
mnt_match = re.compile(r'(?:@[\w_]+)', re.VERBOSE | re.IGNORECASE)
num_match = re.compile(num_reg, re.VERBOSE | re.IGNORECASE)
elongated_match = re.compile(r"(.)\1{2,}")

word_match = re.compile(r'(?:[a-z]+)', re.VERBOSE | re.IGNORECASE)
word2_match = re.compile(r'(?:[\w_]+)', re.VERBOSE | re.IGNORECASE)
word3_match = re.compile(r"(?:[a-z][a-z'\-_]+[a-z])", re.VERBOSE | re.IGNORECASE)


punctuation = [':', '?', '!', '"', '-', "'", '."', ';',
        '.', ',', '(', ')', '&', '@', '#', '%']

stopwords = StopwordsLoader.words() + punctuation

def token_reduce(token):
    return elongated_match.sub(r"\1\1", token)


def tweet_tokenize(tw_text):
    return token_match.findall(tw_text)


def tweet_preprocessing(tw_text):
    tokens = tweet_tokenize(tw_text)
    tokens = [token if emotion_match.search(token) else token.lower() for token in tokens]
    tokens = [token for token in tokens if token not in stopwords and not (num_match.search(token))]

    for i in range(len(tokens)):
        if mnt_match.search(tokens[i]):
            tokens[i] = "usrmnt"

        if url_match.search(tokens[i]):
            tokens[i] = "usrurl"

        if word_match.search(tokens[i]):
            if elongated_match.search(tokens[i]):
                w = Word(token_reduce(tokens[i]))
                tokens[i] = w.spellcheck()[0][0]

    return tokens

def tweet_textarray_preprocessing(file_path):
    a = 0
    b = 0
    c = 0
    print("Start preprocessing plz waite few minute")
    set_file = open(file_path, "r", encoding='UTF-8')
    phrase = []
    labels = []
    t = 0
    for line in set_file:
        line_data = line.split('\t')
        label = line_data[1]
        tw_text = line_data[2]
        tokens = tweet_tokenize(tw_text)
        tokens = [token if emotion_match.search(token) else token.lower() for token in tokens]
        tokens = [token for token in tokens if token not in punctuation and not (num_match.search(token)) and not (emotion_match.search(token))]

        for i in range(len(tokens)):
            if mnt_match.search(tokens[i]):
                tokens[i] = "usrmnt"

            if url_match.search(tokens[i]):
                tokens[i] = "usrurl"
            
            tokens[i] = re.sub(alpha_reg,"",tokens[i])

        tokens = [token for token in tokens if word2_match.search(token) or word3_match.search(token)]
        for i in range(len(tokens)):
            if word_match.search(tokens[i]):
                    if elongated_match.search(tokens[i]):
                        w = Word(token_reduce(tokens[i]))
                        tokens[i] = w.spellcheck()[0][0]

        if(len(tokens) > 0):
            tw_text = " ".join(tokens)
            if label == "negative":
                a+=1
                phrase.append(tw_text)
                labels.append(0)
            elif label == "neutral":
                if b<=8326:
                    b+=1
                    phrase.append(tw_text)
                    labels.append(1)
            elif label == "positive":
                if c<=8326:
                    c+=1
                    phrase.append(tw_text)
                    labels.append(2)            
        t+=1
        if t % 100 == 0:
            print("already processed lines:", t)
    X = np.asarray(phrase)
    Y = np.asarray(labels, dtype=int)
    print("preprocessing end:", a, b, c )
    X.dump("cache/soft_x.txt")
    Y.dump("cache/soft_y.txt")
    return X, Y

def test_tweet_textarray_preprocessing(file_path):
    print("Start preprocessing test data plz waite few minute")
    set_file = open(file_path, "r", encoding='UTF-8')
    phrase = []
    labels = []
    t = 0
    index_set = []
    for line in set_file:
        line_data = line.split('\t')
        label = line_data[1]
        tw_text = line_data[2]
        tokens = tweet_tokenize(tw_text)
        tokens = [token if emotion_match.search(token) else token.lower() for token in tokens]
        tokens = [token for token in tokens if token not in punctuation and not (num_match.search(token)) and not (emotion_match.search(token))]

        for i in range(len(tokens)):
            if mnt_match.search(tokens[i]):
                tokens[i] = "usrmnt"

            if url_match.search(tokens[i]):
                tokens[i] = "usrurl"
            
            tokens[i] = re.sub(alpha_reg,"",tokens[i])

        tokens = [token for token in tokens if word2_match.search(token) or word3_match.search(token)]
        for i in range(len(tokens)):
            if word_match.search(tokens[i]):
                    if elongated_match.search(tokens[i]):
                        w = Word(token_reduce(tokens[i]))
                        tokens[i] = w.spellcheck()[0][0]

        if(len(tokens) > 0):
            index_set.append(line_data[0])
            tw_text = " ".join(tokens)
            phrase.append(tw_text)
            if label == "negative":
                labels.append(0)
            elif label == "neutral":
                labels.append(1)
            elif label == "positive":
                labels.append(2)
            
            t+=1
    X = np.asarray(phrase)
    Y = np.asarray(labels, dtype=int)
    print("preprocessing end")
    return index_set, X, Y


def tweet_word2prob(file_path):
    training_set_file = open(file_path, "r", encoding='UTF-8')
    word2prob = {}
    negative = 0
    neutral  = 0
    positive = 0
    for line in training_set_file:
        line_data = line.split('\t')
        label = line_data[1]
        tw_text = line_data[2]
        row_tokens = tweet_preprocessing(tw_text)
        for token in row_tokens:
            if token in word2prob:
                if label == "negative":
                    word2prob[token][0] += 1
                    negative += 1
                elif label == "neutral":
                    word2prob[token][1] += 1
                    neutral += 1
                elif label == "positive":
                    word2prob[token][2] += 1
                    positive += 1
            else:
                if label == "negative":
                    word2prob[token] = np.array([1,0,0])
                    negative += 1
                elif label == "neutral":
                    word2prob[token] = np.array([0,1,0])
                    neutral += 1
                elif label == "positive":
                    word2prob[token] = np.array([0,0,1])
                    positive += 1
    
    class_freqency = np.array([negative,neutral,positive])

    for word, probs in word2prob.items():
        if sum(probs) != 0:
            word2prob[word] = word2prob[word]/class_freqency
            word2prob[word] = word2prob[word]/sum(word2prob[word])
    
    pickle.dump(word2prob, open("cache/word2prob.dat", "wb"))
    return word2prob

def tweet_bigram2prob(file_path):
    training_set_file = open(file_path, "r", encoding='UTF-8')
    bigram2prob = {}
    negative = 0
    neutral  = 0
    positive = 0
    for line in training_set_file:
        line_data = line.split('\t')
        label = line_data[1]
        tw_text = line_data[2]
        row_tokens = tweet_preprocessing(tw_text)
        row_bigram = bigrams(row_tokens)
        for bigram in row_bigram:
            if bigram in bigram2prob:
                if label == "negative":
                    bigram2prob[bigram][0] += 1
                    negative += 1
                elif label == "neutral":
                    bigram2prob[bigram][1] += 1
                    neutral += 1
                elif label == "positive":
                    bigram2prob[bigram][2] += 1
                    positive += 1
            else:
                if label == "negative":
                    bigram2prob[bigram] = np.array([1,0,0])
                    negative += 1
                elif label == "neutral":
                    bigram2prob[bigram] = np.array([0,1,0])
                    neutral += 1
                elif label == "positive":
                    bigram2prob[bigram] = np.array([0,0,1])
                    positive += 1

    class_freqency = np.array([negative,neutral,positive])
    for bigram, probs in bigram2prob.items():
        if sum(probs) != 0:
            bigram2prob[bigram] = bigram2prob[bigram]/class_freqency
            bigram2prob[bigram] =  bigram2prob[bigram]/sum(bigram2prob[bigram])
    
    pickle.dump(bigram2prob, open("cache/bigram2prob.dat", "wb"))
    return bigram2prob

def tweet_orgnize_word2prob_matrix(file_path, feature_len, is_train = True):
    word2prob = pickle.load(open("cache/word2prob.dat", "rb"))
    training_set_file = open(file_path, "r", encoding='UTF-8')
    x_train_matrix = []
    y_train_matrix = []
    index_set = []
    for line in training_set_file:
        line_data = line.split('\t')
        index = line_data[0]
        label = line_data[1]
        tw_text = line_data[2]
        row_tokens = tweet_preprocessing(tw_text)
        count = 0
        prob_matrix = []
        index_set.append(index)
        for token in row_tokens:
            if token in word2prob:
                prob_matrix.append(word2prob[token])
            else:
                prob_matrix.append(np.array([1.0/3,1.0/3,1.0/3]))
            count+=1

        for i in range(count,feature_len):
            prob_matrix.append(np.array([0,0,0]))
        
        x_train_row = np.array(prob_matrix)
        x_train_row = x_train_row.T.reshape((feature_len*3, )).T
        x_train_matrix.append(x_train_row)
        
        if label == "negative":
            y_train_matrix.append([1,0,0])
        elif label == "neutral":
            y_train_matrix.append([0,1,0])
        elif label == "positive":
            y_train_matrix.append([0,0,1])

    x_train_matrix = np.array(x_train_matrix)
    y_train_matrix = np.array(y_train_matrix)

    if is_train:
        x_train_matrix.dump("cache/train_data_uni.txt")
        y_train_matrix.dump("cache/train_label_uni.txt")
    else:
        return index_set, x_train_matrix, y_train_matrix

def tweet_orgnize_bigram2prob_matrix(file_path, feature_len,is_train = True):
    bigram2prob = pickle.load(open("cache/bigram2prob.dat", "rb"))
    training_set_file = open(file_path, "r", encoding='UTF-8')
    x_train_matrix = []
    y_train_matrix = []
    index_set = []
    for line in training_set_file:
        line_data = line.split('\t')
        index = line_data[0]
        label = line_data[1]
        tw_text = line_data[2]
        row_tokens = tweet_preprocessing(tw_text)
        row_bigram = bigrams(row_tokens)
        count = 0
        prob_matrix = []
        index_set.append(index)
        for bigram in row_bigram:
            if bigram in bigram2prob:
                prob_matrix.append(bigram2prob[bigram])
            else:
                prob_matrix.append(np.array([1.0/3,1.0/3,1.0/3]))
            count+=1
        for i in range(count,feature_len):
            prob_matrix.append(np.array([0,0,0]))
        
        x_train_row = np.array(prob_matrix)
        x_train_row = x_train_row.T.reshape((feature_len*3, )).T
        x_train_matrix.append(x_train_row)
        
        if label == "negative":
            y_train_matrix.append([1,0,0])
        elif label == "neutral":
            y_train_matrix.append([0,1,0])
        elif label == "positive":
            y_train_matrix.append([0,0,1])

    x_train_matrix = np.array(x_train_matrix)
    y_train_matrix = np.array(y_train_matrix)

    if is_train:
        x_train_matrix.dump("cache/train_data_bi.txt")
        y_train_matrix.dump("cache/train_label_bi.txt")
    else:
        return index_set, x_train_matrix, y_train_matrix

# if __name__ == '__main__':
#     bigram2prob = pickle.load(open("cache/bigram2prob.dat", "rb"))
#     for k,v in bigram2prob.items():
#         print(k,v)
