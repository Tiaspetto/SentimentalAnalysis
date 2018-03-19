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


stopwords = [':', '?', '!', '"', '-', "'", '."', ';',
        '.', ',', '(', ')', '&', '@', '#', '%']


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
    a=0
    b=0
    c=0
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
        tokens = [token for token in tokens if token not in stopwords and not (num_match.search(token)) and not (emotion_match.search(token))]

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

            X = np.asarray(phrase)
            Y = np.asarray(labels, dtype=int)
            
            t+=1
            if t % 100 == 0:
                print("already processed lines:", t)
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
        tokens = [token for token in tokens if token not in stopwords and not (num_match.search(token)) and not (emotion_match.search(token))]

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

            X = np.asarray(phrase)
            Y = np.asarray(labels, dtype=int)
            
            t+=1
            # if t % 100 == 0:
            #     print("already processed lines:", t)
    print("preprocessing end")
    return index_set, X, Y



def tweet_preprocessing_train_unigram(file_path, feature_len):
    token_set = []
    training_set_tokens = []
    training_set_ids = []
    training_set_labels = []

    training_set_file = open(file_path, "r", encoding='UTF-8')

    print("Preporcessing Twitter data, please waite few minute")
    for line in training_set_file:
        line_data = line.split('\t')
        label = line_data[1]
        tw_text = line_data[2]
        row_tokens = tweet_preprocessing(tw_text)
        token_set.extend(row_tokens)
        training_set_tokens.append(row_tokens)
        y = [0, 0, 0]
        if label == "negative":
            y = [1, 0, 0]
        elif label == "neutral":
            y = [0, 1, 0]
        elif label == "positive":
            y = [0, 0, 1]
        training_set_labels.append(y)
    token_set = list(set(token_set))

    word2ids = {}
    for index, word in enumerate(token_set):
        word2ids[word] = index + 1

    for row in training_set_tokens:
        x = np.zeros(feature_len)
        for ix, word in enumerate(row):
            if (ix >= feature_len):
                break
            x[ix] = word2ids[word]
        training_set_ids.append(x)

    x_train_matrix = np.array(training_set_ids)/len(token_set)
    y_train_matrix = np.array(training_set_labels)

    x_train_matrix.dump("cache/train_data_uni.txt")
    y_train_matrix.dump("cache/train_label_uni.txt")

    pickle.dump(word2ids, open("cache/word2ids.dat", "wb"))

def dev_tweet_preprocessing_train_unigram(file_path, feature_len, word2ids):
    token_set = []
    training_set_tokens = []
    training_set_ids = []
    training_set_labels = []

    training_set_file = open(file_path, "r", encoding='UTF-8')

    print("Preporcessing Twitter data, please waite few minute")
    for line in training_set_file:
        line_data = line.split('\t')
        label = line_data[1]
        tw_text = line_data[2]
        row_tokens = tweet_preprocessing(tw_text)
        token_set.extend(row_tokens)
        training_set_tokens.append(row_tokens)
        y = [0, 0, 0]
        if label == "negative":
            y = [1, 0, 0]
        elif label == "neutral":
            y = [0, 1, 0]
        elif label == "positive":
            y = [0, 0, 1]
        training_set_labels.append(y)

    token_set = list(set(token_set))

    for index, word in enumerate(token_set):
        if word not in word2ids:
            word2ids[word] = index + 1

    for row in training_set_tokens:
        x = np.zeros(feature_len)
        for ix, word in enumerate(row):
            if (ix >= feature_len):
                break
            x[ix] = word2ids[word]
        training_set_ids.append(x)

    x_train_matrix = np.array(training_set_ids)/len(word2ids)
    y_train_matrix = np.array(training_set_labels)

    pickle.dump(word2ids, open("cache/word2ids.dat", "wb"))

    return x_train_matrix, y_train_matrix


def tweet_preprocessing_test_unigram(file_path, feature_len, word2ids):
    print("Preporcessing test Twitter data, please waite few minute...")
    test_set_file = open(file_path, "r", encoding='UTF-8')

    test_set_tokens = []
    test_set_ids = []
    test_set_labels = []
    
    index_set = []
    for line in test_set_file:
        line_data = line.split('\t')
        label = line_data[1]
        tw_text = line_data[2]
        index_set.append(line_data[0])
        row_tokens = tweet_preprocessing(tw_text)
        test_set_tokens.append(row_tokens)
        y = [0, 0, 0]
        if label == "negative":
            y = [1, 0, 0]
        elif label == "neutral":
            y = [0, 1, 0]
        elif label == "positive":
            y = [0, 0, 1]
        test_set_labels.append(y)

    for row in test_set_tokens:
        x = np.zeros(feature_len)
        for ix, word in enumerate(row):
            if (ix >= feature_len):
                break
            if word in word2ids:
                x[ix] = word2ids[word]
            else:
                x[ix] = 0
        test_set_ids.append(x)

    x_test_matrix = np.array(test_set_ids)/len(word2ids)
    y_test_matrix = np.array(test_set_labels)

    x_test_matrix.dump("cache/test_data_uni.txt")
    y_test_matrix.dump("cache/test_label_uni.txt")

    return index_set


