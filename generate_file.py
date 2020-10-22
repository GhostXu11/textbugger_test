import pickle
import json
import random
import time

import nltk
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from textbugger_utils import transform_to_feature_vector
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression


# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences


# my_matrix = pd.read_csv("data/IMDB_Dataset.csv")
# my_matrix = my_matrix[:2000]
# train, test = train_test_split(my_matrix, test_size=0.2)
# train.to_csv("data/IMDB/train_IMDB.csv")
# test.to_csv("data/IMDB/test_IMDB.csv")


# nltk.download('punkt')


def get_score(row):
    if row['sentiment'] == "positive":
        return 1
    return 0


def get_train_IMDB():
    df = pd.read_csv("data/IMDB/train_IMDB.csv")

    df['score'] = df.apply(lambda row: get_score(row), axis=1)
    df = df[['review', 'score']]

    data = {'train': {'pos': [], 'neg': []}, 'test': {'pos': [], 'neg': []}}
    for index, row in df.iterrows():
        if row['score'] == 1:
            data['train']['pos'].append(row['review'])
        else:
            data['train']['neg'].append(row['review'])

    # pickle.dump(data, open("data/IMDB/IMDB_train_tokens.p", "wb"))
    print(data)


# get_train_IMDB()


def get_test_token_IMDB():
    df = pd.read_csv("data/IMDB/test_IMDB.csv")

    df['score'] = df.apply(lambda row: get_score(row), axis=1)
    df = df[['review', 'score']]

    data = pickle.load(open("data/IMDB/IMDB_train_tokens.p", "rb"))
    for index, row in df.iterrows():
        if row['score'] == 1:
            data['test']['pos'].append(row['review'])
        else:
            data['test']['neg'].append(row['review'])

    pickle.dump(data, open("data/IMDB/IMDB_sentences_tokens.p", "wb"))


# get_test_token_IMDB()


def get_tokens_IMDB():
    data = pickle.load(open("data/IMDB/IMDB_sentences_tokens.p", "rb"))

    res = {'train': {'pos': [], 'neg': []}, 'test': {'pos': [], 'neg': []}}

    for key1 in data:
        for key2 in data[key1]:
            for sentence in data[key1][key2]:
                res[key1][key2].append(nltk.word_tokenize(sentence))
    pickle.dump(res, open("data/IMDB/IMDB_tokens.p", "wb"))
    # print(res)


# get_tokens_IMDB()


def get_glove_IMDB():
    start = time.time()
    with open('data/glove.6B.50d.json') as f:
        glove_vectors = json.load(f)
    end = time.time()
    print("DONE LOADING: {} minutes".format((end - start) / 60))

    res = {}
    num_in = 0
    num_out = 0
    data = pickle.load(open("data/IMDB/IMDB_tokens.p", "rb"))
    for key1 in data:
        for key2 in data[key1]:
            for token_list in data[key1][key2]:
                for word in token_list:
                    if word in glove_vectors:
                        res[word] = glove_vectors[word]
                        num_in += 1
                    else:
                        res[word] = [(random.random() / 5) - 0.1 for i in range(300)]
                        num_out += 1

    with open('data/IMDB/glove_imdb.json', 'w') as outfile:
        json.dump(res, outfile)


def create_IMDB_vector():
    glove_vectors = json.load(open("data/glove.6B.50d.json", "rb"))
    data = pickle.load(open("data/IMDB/IMDB_tokens.p", "rb"))

    res = {}
    for key1 in data:
        res[key1] = {}
        for key2 in data[key1]:
            res[key1][key2] = []

    print(res)

    for key1 in data:
        for key2 in data[key1]:
            vects = []
            for doc in data[key1][key2]:
                vect = transform_to_feature_vector(doc, glove_vectors)
                vects.append(vect)
            res[key1][key2] = vects

    pickle.dump(res, open("data/IMDB/IMDB_vectors.p", "wb"))


# create_IMDB_vector()


def make_LR(dataset):
    data = pickle.load(open("data/{}/{}_vectors.p".format(dataset, dataset), "rb"))

    ## Train
    x_train_pos = data['train']['pos']
    x_train_neg = data['train']['neg']
    x_train = []
    x_train.extend(x_train_pos)
    x_train.extend(x_train_neg)
    y_train = [1 for i in range(len(x_train_pos))]
    y_train.extend([0 for i in range(len(x_train_neg))])
    x_train = np.array(x_train, dtype='float')
    y_train = np.array(y_train, dtype='float')
    print(x_train.shape, y_train.shape)
    x_train = x_train.reshape((1600, 300))
    print(x_train.shape, y_train.shape)

    ## Test
    x_test_pos = data['test']['pos']
    x_test_neg = data['test']['neg']
    x_test = []
    x_test.extend(x_test_pos)
    x_test.extend(x_test_neg)
    y_test = [1 for i in range(len(x_test_pos))]
    y_test.extend([0 for i in range(len(x_test_neg))])

    x_test = np.array(x_test, dtype='float')
    y_test = np.array(y_test, dtype='float')
    x_test = x_test.reshape((400, 300))
    print(x_test.shape, y_test.shape)

    ## Shuffle
    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)

    model = LogisticRegression(random_state=42).fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    print(acc)

    pickle.dump(model, open("models/LR/LR_{}.p".format(dataset), "wb"))


make_LR('IMDB')

def featureExtraction_IMDB_Train():
    print("Loading...")
    with open("data/glove.6B.50d.json") as json_file:
        glove_vectors = json.load(json_file)
    print("DONE LOADING GLOVE")

    with open("data/IMDB/IMDB_tokens.p", 'rb') as fp:
        data = pickle.load(fp)

    print("DONE LOADING IMDB")

    for key1 in data:
        for key2 in data[key1]:
            feature_vectors = []
            for point in data[key1][key2]:
                vector = getFeatureVector(point, glove_vectors)
                feature_vectors.append(vector)

            data[key1][key2] = feature_vectors

    with open('data/IMDB/IMDB_vectors_new.p', 'wb') as fp:
        pickle.dump(data, fp)
    print("Done")


def getFeatureVector(point, glove_vectors):
    numTokens = 0
    vect = np.asarray([0] * 300)
    for token in point:
        if token in glove_vectors:
            vect = np.add(vect, np.asarray(glove_vectors[token]))
            numTokens += 1
    np.seterr(divide='ignore', invalid='ignore')
    vect = np.divide(vect, numTokens)
    return vect


# featureExtraction_IMDB_Train()
