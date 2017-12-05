# -*- coding: utf-8 -*-
__author__ = 'Aigul'


import json
import pandas as pd


emoji_dict = {'sun':1, 'music':2}
def preproc(line):
    import re
    line = re.sub('\n', ' ', line)
    line = re.sub('\t', ' ', line)
    return line
def read_json(df, emoji_name):
    with open('search_'+emoji_name+'.json', encoding='utf-8') as f_in:
        data = json.load(f_in)
    for tweet_id in data:
        text = preproc(data[tweet_id]['text'])
        if text:
            df['id'].append(tweet_id)
            df['text'].append(text)
            df['sent'].append(emoji_dict[emoji_name])
    return df

df = {'text':[], 'id':[], 'sent':[]}
df = read_json(df, 'sun')
df = read_json(df, 'music')

df = pd.DataFrame(df)
df.to_csv('t.csv', index=False, sep='\t', encoding='utf8')


