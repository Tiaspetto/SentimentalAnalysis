#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import testsets
import evaluation
from preprocessing import *
import os
import pickle
from plaindeepneural import *
import numpy as np
from sentence_label import *
from LSTMKeras import *
from keras.models import load_model
# TODO: load training data
"""
unigram
"""
LSTM_maxLen = 0
dev_path = "semeval-tweets/twitter-dev-data.txt"
train_path = "semeval-tweets/twitter-training-data.txt"
test_path = "semeval-tweets/"
if not os.path.exists("cache/train_data_uni.txt") or not os.path.exists("cache/train_label_uni.txt"):
    tweet_word2prob(train_path)
    tweet_orgnize_word2prob_matrix(train_path, 140)

if not os.path.exists("cache/train_data_bi.txt") or not os.path.exists("cache/train_label_bi.txt"):
    tweet_bigram2prob(train_path)
    tweet_orgnize_bigram2prob_matrix(train_path, 140)
print("Loading glove twitter data word2vec, please waite few second...")
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('semeval-tweets/glove.twitter.27B.50d.txt')

# You may rename the names of the classifiers to something more descriptive
for classifier in ['myclassifier1', 'myclassifier2', 'myclassifier3']:
    if classifier == 'myclassifier1':
        print('Training ' + classifier)
        x_train_matrix = np.load("cache/train_data_uni.txt")
        y_train_matrix = np.load("cache/train_label_uni.txt")
        if not os.path.exists("cache/plain_model.dat"):
            parameters = L_layer_model(x_train_matrix.T, y_train_matrix.T, [420,140,40,20,3])
            pickle.dump(parameters, open("cache/plain_model.dat", "wb"))

        parameters = pickle.load(open("cache/plain_model.dat", "rb"))
        NN_predict(x_train_matrix.T, y_train_matrix.T, parameters)
        


    elif classifier == 'myclassifier2':
        print('Training ' + classifier)
        x_train_matrix = np.load("cache/train_data_bi.txt")
        y_train_matrix = np.load("cache/train_label_bi.txt")
        if not os.path.exists("cache/plain_model_bi.dat"):
            parameters = L_layer_model(x_train_matrix.T, y_train_matrix.T, [420,140,40,20,3])
            pickle.dump(parameters, open("cache/plain_model_bi.dat", "wb"))

        parameters = pickle.load(open("cache/plain_model_bi.dat", "rb"))
        NN_predict(x_train_matrix.T, y_train_matrix.T, parameters)
        
    elif classifier == 'myclassifier3':
        if not os.path.exists("cache/soft_x.txt") or not os.path.exists("cache/soft_y.txt"):
            tweet_textarray_preprocessing(train_path)
        X_train = np.load("cache/soft_x.txt")
        Y_train = np.load("cache/soft_y.txt")

        #X_train, Y_train = read_csv()
        LSTM_maxLen = len(max(X_train, key=len).split())+1
        if not os.path.exists("cache/LSTMKeras.dat"):
            Y_train_oh = convert_to_one_hot(Y_train, C = 3)
            model = LSTMKeras_model((LSTM_maxLen,), word_to_vec_map, word_to_index)
            model.summary()
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            X_train_indices = sentences_to_indices(X_train, word_to_index, LSTM_maxLen)
            model.fit(X_train_indices, Y_train_oh, epochs = 20, batch_size = 32, shuffle=True)
            model.save("cache/LSTMKeras.dat")
        print("LSTMKeras is a huge model, so if you have already load trained model, the predictions on training set will not be shown\n")

    for testset in testsets.testsets:
        sub_test_path = test_path + testset
        if classifier == 'myclassifier1':
            index_set, x_test_matrix, y_test_matrix = tweet_orgnize_word2prob_matrix(sub_test_path, 140, False)
            parameters = pickle.load(open("cache/plain_model.dat", "rb"))
            predicted_label = NN_predict(x_test_matrix.T, y_test_matrix.T, parameters)
            predictions = {}
            for i in range(len(index_set)):
                if predicted_label[i].index(max(predicted_label[i])) == 0:
                    predictions[index_set[i]] = 'negative'
                elif predicted_label[i].index(max(predicted_label[i])) == 1:
                    predictions[index_set[i]] = 'neutral'
                elif predicted_label[i].index(max(predicted_label[i])) == 2:
                    predictions[index_set[i]] = 'positive'
            evaluation.evaluate(predictions, sub_test_path, classifier)
            evaluation.confusion(predictions, sub_test_path, classifier)

        if classifier == 'myclassifier2':
            index_set, x_test_matrix, y_test_matrix = tweet_orgnize_bigram2prob_matrix(sub_test_path, 140, False)
            parameters = pickle.load(open("cache/plain_model_bi.dat", "rb"))
            predicted_label = NN_predict(x_test_matrix.T, y_test_matrix.T, parameters)
            predictions = {}
            for i in range(len(index_set)):
                if predicted_label[i].index(max(predicted_label[i])) == 0:
                    predictions[index_set[i]] = 'negative'
                elif predicted_label[i].index(max(predicted_label[i])) == 1:
                    predictions[index_set[i]] = 'neutral'
                elif predicted_label[i].index(max(predicted_label[i])) == 2:
                    predictions[index_set[i]] = 'positive'
            evaluation.evaluate(predictions, sub_test_path, classifier)
            evaluation.confusion(predictions, sub_test_path, classifier)

        if classifier == "myclassifier3":
            model = load_model("cache/LSTMKeras.dat")
            index_set, X_test, Y_test = test_tweet_textarray_preprocessing(sub_test_path)
            X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = LSTM_maxLen)
            Y_test_oh = convert_to_one_hot(Y_test, C = 3)
            loss, acc = model.evaluate(X_test_indices, Y_test_oh)
            pred = model.predict(X_test_indices)
            predictions = {}
            print(acc)
            for i in range(len(index_set)):
                 num = np.argmax(pred[i])
                 if num == 0:
                     predictions[index_set[i]] = 'negative'
                 elif num == 1:
                     predictions[index_set[i]] = 'neutral'
                 elif num == 2:
                     predictions[index_set[i]] = 'positive'

            evaluation.evaluate(predictions, sub_test_path, classifier)
            evaluation.confusion(predictions, sub_test_path, classifier)

    
