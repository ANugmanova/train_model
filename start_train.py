# -*- coding: utf-8 -*-
"""Finetuning example.

Trains the DeepMoji model on the SS-Youtube dataset, using the 'last'
finetuning method and the accuracy metric.

The 'last' method does the following:
0) Load all weights except for the softmax layer. Do not add tokens to the
   vocabulary and do not extend the embedding layer.
1) Freeze all layers except for the softmax layer.
2) Train.
"""

from __future__ import print_function
import pandas as pd
from model_def import deepmoji_transfer
from finetuning import (
     calculate_batchsize_maxlen,
     finetune)
from sentence_tokenizer import SentenceTokenizer

import re

def load_benchmark(data, vocab, extend_with=0):

    # Decode data
    try:
        texts = [x for x in data['text']]
    except UnicodeDecodeError:
        texts = [x for x in data['text']]

    # Extract labels
    labels = [x for x in data['sent']]

    batch_size, maxlen = calculate_batchsize_maxlen(texts)

    st = SentenceTokenizer(vocab, maxlen)

    # Split up dataset. Extend the existing vocabulary with up to extend_with
    # tokens from the training dataset.
    texts, labels, added = st.split_train_val_test(texts,
                                                   labels,
                                                   extend_with=extend_with)
    return {'texts': texts,
            'labels': labels,
            'added': added,
            'batch_size': batch_size,
            'maxlen': maxlen}


def load_vocab(data):
    vocab = {}
    for raw in data['text']:
        temp = re.sub('[^a-zA-Zа-яА-Я ]', '', raw)
        temp = temp.lower()
        for w in temp.split():
            if w not in vocab:
                vocab[w] = len(vocab)
    # print(vocab)
    vocab['newword'] = len(vocab)
    return vocab


if __name__ == "__main__":

    DATASET_PATH = './data_tweets/t.csv' #t.csv'
    nb_classes = 2
    delete_non_raws = False


    if delete_non_raws:
        data = pd.read_csv(DATASET_PATH, sep='\t')
        data = data.dropna()
        data.to_csv(DATASET_PATH, sep='\t', index=False)

    data = pd.read_csv(DATASET_PATH, sep='\t')
    data = data[:5000]

    vocab = load_vocab(data)

    print(len(vocab))

    # Load dataset
    data_tr_cv_ts = load_benchmark(data, vocab)

    # Set up model and finetune
    # init
    model = deepmoji_transfer(nb_classes, data_tr_cv_ts['maxlen'])  # from PRETRAINED_PATH load model
    # print_layer_summary
    model.summary()
    #
    print(len(data_tr_cv_ts['texts'][0][0]))
    model, acc = finetune(model, data_tr_cv_ts['texts'], data_tr_cv_ts['labels'], nb_classes,
                          data_tr_cv_ts['batch_size'], method='last',
                          epoch_size=5000, nb_epochs=1000, verbose=5)#'last')
    print('Acc: {}'.format(acc))
