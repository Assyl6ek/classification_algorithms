from django.core import serializers
from django.http import JsonResponse
from django.shortcuts import render
from rest_framework.decorators import api_view
import pickle
from algorithm.models import comment
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import json

# Create your views here.

@api_view(['POST'])
def classify(request):
    saved_vocab = open("savedVocab.pkl", "rb")
    vocab = pickle.load(saved_vocab)

    df2 = pd.DataFrame(request.data['data'])

    corp = []
    for i in range(len(df2.index)):
        report = re.sub("[^А-я]", " ", str(df2['text'][i]))
        report = report.lower()
        report = report.split()
        ps = PorterStemmer()
        report = [ps.stem(word) for word in report if word not in set(stopwords.words("russian"))]
        report = " ".join(report)
        corp.append(report)

    w = len(vocab)
    s = len(corp)
    matrix = np.zeros((s, w))

    for i in range(s):
        sen_line = corp[i].split(" ")
        for feature in sen_line:
            if feature != '' and feature in vocab.keys():
                matrix[i][vocab[feature]] += 1

    tensor = df2.copy()
    tensor['text'] = matrix.tolist()
    list_of_lists = tensor['text'].values.tolist()
    x_set = tf.constant(list_of_lists)

    reconstructed_model = keras.models.load_model("savedModel")
    predictions = reconstructed_model.predict(x_set, batch_size=None, verbose=0, steps=None)
    predictions = np.argmax(predictions.T, axis=0)
    print('prediction: ', predictions)

    df2['is_positive'] = predictions

    positive_features = {
        "еда": 0,
        "сервис": 0,
        "доставка": 0,
        "интерьер": 0,
        "атмосфера": 0,
        "музыка": 0,
        "месторасположение": 0,
    }
    negative_features = {
        "еда": 0,
        "сервис": 0,
        "доставка": 0,
        "интерьер": 0,
        "атмосфера": 0,
        "музыка": 0,
        "месторасположение": 0,
    }

    df_positive = df2.loc[df2['is_positive'] == 1]
    df_negative = df2.loc[df2['is_positive'] == 0]

    positive_corpus = []
    for i in range(len(df_positive.index)):
        report = re.sub("[^А-я]", " ", str(df_positive["text"][i]))
        report = report.lower()
        report = report.split()
        ps = PorterStemmer()
        report = [ps.stem(word) for word in report if not word in set(stopwords.words("russian"))]
        report = " ".join(report)
        positive_corpus.append(report)

    negative_corpus = []
    for i in range(len(df_negative.index)):
        report = re.sub("[^А-я]", " ", str(df_negative["text"][i]))
        report = report.lower()
        report = report.split()
        ps = PorterStemmer()
        report = [ps.stem(word) for word in report if not word in set(stopwords.words("russian"))]
        report = " ".join(report)
        negative_corpus.append(report)

    f = len(positive_corpus)
    for i in range(f):
        sen_line = positive_corpus[i].split(" ")
        for word in sen_line:
            if word != '' and word in positive_features.keys():
                positive_features[word] += 1

    n = len(negative_corpus)
    for i in range(n):
        sen_line = negative_corpus[i].split(" ")
        for word in sen_line:
            if word != '' and word in negative_features.keys():
                negative_features[word] += 1

    advantages = []
    drawbacks = []

    for val in positive_features.keys():
        if positive_features[val] - negative_features[val] > 0:
            advantages.append(val)
        elif positive_features[val] - negative_features[val] < 0:
            drawbacks.append(val)

    print("Достоинства: ", advantages)
    print("Недостатки: ", drawbacks)

    request.data['advantages'] = advantages
    request.data['disadvantages'] = drawbacks
    count = 0
    for com in request.data['data']:
        com['is_positive'] = int(predictions[count])
        count += 1

    print('response Converted: ', request.data)
    return JsonResponse(request.data, safe=False)
