import numpy as np
import json
import pickle
import os
from sklearn.model_selection import train_test_split
import nltk
import time
import pprint
from textbugger_utils import get_prediction_given_tokens
from whitebox import WhiteBox
import random


def testWhiteBox(data_type, model_type, num_samples):
    start = time.time()
    ## Import glove-vectors once

    glove_vectors = json.load(open("data/glove.6B.50d.json".format(data_type, data_type), "rb"))

    embed_map = pickle.load(open("data/embed_map.p".format(data_type, data_type), "rb"))

    ## Get Dataset
    if (data_type == 'IMDB'):
        data = pickle.load(open("data/IMDB/IMDB_tokens.p", "rb"))

    ## Get Model
    if (model_type == 'LR'):
        if (data_type == 'IMDB'):
            model = pickle.load(open("models/LR/LR_IMDB.p", "rb"))

    # ---- DONE LOADING ----------
    end = time.time()
    print("DONE LOADING: {} minutes".format(np.round((end - start) / 60), 4))

    num_successes = 0
    sample_id = 1
    percent_perturbed = []

    pos_samples = data['test']['pos'][0:num_samples]
    neg_samples = data['test']['neg'][0:num_samples]

    for doc in pos_samples:
        # print(sample_id)
        sample_id += 1
        y = get_prediction_given_tokens(model_type, model, doc, glove_vectors=glove_vectors, embed_map=embed_map,
                                        dataset=data_type)
        y = np.round(y, 0)
        whitebox = WhiteBox(doc, y, model, 0.8, model_type, glove_vectors, embed_map, data_type)
        res = whitebox.whiteBoxAttack()
        if res != None:
            num_successes += 1
            percent_perturbed.append(res[1])
            # print("Successful adversary. Fraction of original input perturbed: {}".format(np.round(res[1],2)))

    for doc in neg_samples:
        # print(sample_id)
        sample_id += 1
        y = get_prediction_given_tokens(model_type, model, doc, glove_vectors=glove_vectors, embed_map=embed_map,
                                        dataset=data_type)
        y = np.round(y, 0)
        whitebox = WhiteBox(doc, y, model, 0.8, model_type, glove_vectors, embed_map, data_type)
        res = whitebox.whiteBoxAttack()
        if res != None:
            num_successes += 1
            percent_perturbed.append(res[1])
            # print("Successful adversary. Fraction of original input perturbed: {}".format(np.round(res[1],2)))

    total_docs = 2 * num_samples
    success_rate = np.round((num_successes / total_docs) * 100, 3)
    perturb_rate = np.round(np.mean(percent_perturbed) * 100, 3)
    # print("{} successful adversaries out of {} total documents. Success rate = {}".format(num_successes,total_docs,success_rate))
    print("Avg % Perturbed: {}".format(perturb_rate))
    print("{} | {} | {}".format(data_type, model_type, success_rate))


testWhiteBox('IMDB','LR', 3)