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
import gensim
# TODO: load training data
"""
unigram
"""
train_path = "semeval-tweets/twitter-training-data.txt"
test_path = "semeval-tweets/twitter-test1.txt"
if not os.path.exists("cache/train_data_uni.txt") or not os.path.exists("cache/train_label_uni.txt"):
    tweet_preprocessing_train_unigram(train_path, 140)

if not os.path.exists("cache/test_data_uni.txt") or not os.path.exists("cache/test_label_uni.txt"):
    if not os.path.exists("cache/word2ids.dat"):
        tweet_preprocessing_train_unigram(train_path, 140)

    word2ids = pickle.load(open("cache/word2ids.dat", "rb"))
    tweet_preprocessing_test_unigram(test_path, 140, word2ids)

# You may rename the names of the classifiers to something more descriptive
for classifier in ['', '', 'myclassifier3']:
    if classifier == 'myclassifier1':
        print('Training ' + classifier)
        x_train_matrix = np.load("cache/train_data_uni.txt")
        y_train_matrix = np.load("cache/train_label_uni.txt")
        if not os.path.exists("cache/plain_model.dat"):
            parameters = L_layer_model(x_train_matrix.T, y_train_matrix.T, [140,40,20,7,3])
            pickle.dump(parameters, open("cache/plain_model.dat", "wb"))

        parameters = pickle.load(open("cache/plain_model.dat", "rb"))
        NN_predict(x_train_matrix.T, y_train_matrix.T, parameters)

    # elif classifier == 'myclassifier2':
        # if not os.path.exists("cache/soft_x.txt") or not os.path.exists("cache/soft_y.txt"):
        #     tweet_textarray_preprocessing(train_path)
        # X_train = np.load("cache/soft_x.txt")
        # Y_train = np.load("cache/soft_y.txt")
        # maxLen = len(max(X_train, key=len).split())
        # Y_oh_train = convert_to_one_hot(Y_train, C = 3)
        # word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('datasets/glove.6B.50d.txt')
        # print(word_to_vec_map['hi'])
    
        # if not os.path.exists("cache/softmax_W.dat") or not os.path.exists("cache/softmax_b.dat"):
        #     pred, W, b = model(X_train, Y_train, word_to_vec_map)
        #     pickle.dump(W, open("cache/softmax_W.dat", "wb"))
        #     pickle.dump(b, open("cache/softmax_b.dat", "wb"))

        # W =  pickle.load(open("cache/softmax_W.dat", "rb"))  
        # b =  pickle.load(open("cache/softmax_b.dat", "rb"))
        # pred_train = predict(X_train, Y_train, W, b, word_to_vec_map)
    elif classifier == 'myclassifier3':
        print('Training ' + classifier)
        if not os.path.exists("cache/soft_x.txt") or not os.path.exists("cache/soft_y.txt"):
            tweet_textarray_preprocessing(train_path)
        X_train = np.load("cache/soft_x.txt")
        Y_train = np.load("cache/soft_y.txt")
        token_set = pickle.load(open("cache/token_set.dat", "rb"))

        # #X_train, Y_train = read_csv()
        # maxLen = len(max(X_train, key=len).split())
        # Y_train_oh = convert_to_one_hot(Y_train, C = 3)
        # word_to_index, index_to_word = index_word_mapping(token_set)
        # word_to_vec_map = gensim.models.Word2Vec.load('cache/embedding')
        # print(word_to_vec_map["zquadwantszayntosmile"])
        #model = LSTMKeras_model((maxLen,), word_to_vec_map, word_to_index)
        # model.summary()
        # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
        # model.fit(X_train_indices, Y_train_oh, epochs = 8, batch_size = 16, shuffle=True)
        # pickle.dump(model, open("cache/LSTMKeras.dat", "wb"))

    # # for testset in testsets.testsets:
    #     # TODO: classify tweets in test set

    #     predictions = {'163361196206957578': 'neutral', '768006053969268950': 'neutral', '742616104384772304': 'neutral', '102313285628711403': 'neutral',
    #                    '653274888624828198': 'neutral'}  # TODO: Remove this line, 'predictions' should be populated with the outputs of your classifier
    #     evaluation.evaluate(predictions, testset, classifier)

    #     evaluation.confusion(predictions, testset, classifier)
