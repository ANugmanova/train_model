# -*- coding: utf-8 -*-
__author__ = 'Aigul'


import json
import pandas as pd
import re

emoji_dict = {'sun': 0, 'music': 1, "laugh": 2, "cry": 3, "smile": 4,
              "sad": 5, "angry": 6, "heart": 7, "broken_heart": 8, "namaste": 9}


def preproc(line):
    import re
    line = re.sub('\n', ' ', line)
    line = re.sub('\t', ' ', line)
    return line


def read_json(df, emoji_name, count = 10000000):
    with open('search_'+emoji_name+'.json', encoding='utf-8') as f_in:
        data = json.load(f_in)
    for i, tweet_id in enumerate(data):
        if i > count:
            break
        text = preproc(data[tweet_id]['text'])
        if text:
            df['id'].append(tweet_id)
            df['text'].append(re.sub('\n', ' ', re.sub('\t', ' ', text)))
            df['sent'].append(emoji_dict[emoji_name])
    return df


if __name__ == '__main__':
    df = {'text':[], 'id':[], 'sent':[]}
    df = read_json(df, 'sun', count=1500)
    df = read_json(df, 'music', count=1500)
    # df = read_json(df, 'laugh', count=1500)
    # df = read_json(df, 'cry', count=1500)
    # df = read_json(df, 'sad', count=1500)
    # df = read_json(df, 'angry', count=1500)
    # df = read_json(df, 'heart', count=1500)
    # df = read_json(df, 'broken_heart', count=1500)
    # df = read_json(df, 'namaste', count=1500)


    df = pd.DataFrame(df)
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv('t.csv', index=False, sep='\t', encoding='utf8')


