import os
import re
import numpy as np
import pandas as pd
import torch
import preprocessing as p
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from torch.utils.data.dataset import Dataset

from params import *


def read_files(data_path):
    all_files = []
    for file in os.listdir(data_path):
        f = pd.read_csv(os.path.join(data_path, file), sep='\t', header=None)
        all_files.append(f)
    full_dataset = pd.concat(all_files, axis=0)[[1, 2]]
    full_dataset.columns = ['category', 'tweet']

    return full_dataset


def init_tokenizer():
    return SocialTokenizer(lowercase=False).tokenize



def pad_sequences(labels, tokenized_tweets, dictionary):

    tokenized_tweets_int = []
    for t in tokenized_tweets:
        tmp = []
        for w in t:
            try:
                tmp.append(dictionary[w])
            except KeyError:
                tmp.append(dictionary['UNK'])
        tokenized_tweets_int.append(tmp)
    # Calculate length of contexts and pad sequences with 0
    lengths = torch.LongTensor([len(x) for x in tokenized_tweets_int])

    tweet_tensor = torch.zeros((len(tokenized_tweets), lengths.max())).long()
    for idx, (seq, seqlen) in enumerate(zip(tokenized_tweets_int, lengths)):
        tweet_tensor[idx, :seqlen] = torch.LongTensor(seq)

    lengths, perm_idx_context = lengths.sort(0, descending=True)
    tweet_tensor = tweet_tensor[perm_idx_context]

    label_tensor = torch.LongTensor(labels)
    label_tensor = label_tensor[perm_idx_context]

    return {'tweet_tensor': tweet_tensor,
            'label_tensor': label_tensor,
            'length': lengths}


class ReaderDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):

        return  {'tokenized_tweet': self.examples.loc[index]['tweet'],
                 'label': self.examples.loc[index]['category']}



