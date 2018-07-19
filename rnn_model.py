import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

from preprocessing import init_tokenizer

def load_glove(embed_file):
    f = open(embed_file,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding

    return model

def tokenize_tweets(train):
    tokenized_tweets = []
    text_processor = TextPreProcessor(
        # terms that will be normalized
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
                   'time', 'url', 'date', 'number'],
        # terms that will be annotated
        annotate={"hashtag", "allcaps", "elongated", "repeated",
                  'emphasis', 'censored'},
        fix_html=True,  # fix HTML tokens

        # corpus from which the word statistics are going to be used
        # for word segmentation
        segmenter="twitter",

        unpack_hashtags=True,  # perform word segmentation on hashtags
        unpack_contractions=True,  # Unpack contractions (can't -> can not)
        spell_correct_elong=False,  # spell correction for elongated words
        tokenizer=SocialTokenizer(lowercase=True).tokenize,


        # list of dictionaries, for replacing tokens extracted from the text,
        # with other expressions. You can pass more than one dictionaries.
        dicts=[emoticons]
    )

    social_tokenizer = init_tokenizer()
    for tweet in train['tweet']:
        if type(tweet)!='str':
            tweet = str(tweet)
        tokenized_tweets.append(social_tokenizer(" ".join(text_processor.pre_process_doc(tweet))))

    return tokenized_tweets

def tokenize_tweets2(train):
    tokenized_tweets = []
    text_processor = TextPreProcessor(
        # terms that will be normalized
        normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
                   'time', 'url', 'date', 'number'],
        # terms that will be annotated
        annotate={"hashtag", "allcaps", "elongated", "repeated",
                  'emphasis', 'censored'},
        fix_html=True,  # fix HTML tokens

        # corpus from which the word statistics are going to be used
        # for word segmentation
        segmenter="twitter",

        unpack_hashtags=True,  # perform word segmentation on hashtags
        unpack_contractions=True,  # Unpack contractions (can't -> can not)
        spell_correct_elong=False,  # spell correction for elongated words
        tokenizer=SocialTokenizer(lowercase=True).tokenize,


        # list of dictionaries, for replacing tokens extracted from the text,
        # with other expressions. You can pass more than one dictionaries.
        dicts=[emoticons]
    )

    social_tokenizer = init_tokenizer()
    for tweet in train:
        if type(tweet)!='str':
            tweet = str(tweet)
        tokenized_tweets.append(social_tokenizer(" ".join(text_processor.pre_process_doc(tweet))))

    return tokenized_tweets

def create_vocab(tokenized_tweet, glove):

    unique_words = list(set([w for word in tokenized_tweet for w in word]))

    matrix_len = len(unique_words)
    weights_matrix = np.zeros((matrix_len, 300))
    dictionary = {}

    for i, word in enumerate(unique_words):
        try:
            dictionary[word] = i
            weights_matrix[i] = glove[word]
        except KeyError:
            dictionary['UNK'] = i
            weights_matrix[i] = np.zeros(300)

    return weights_matrix, dictionary


class RnnModel(nn.Module):
    def __init__(self, params):
        nn.Module.__init__(self)

        self.embed = nn.Embedding(params['n_embed'], params['dim_embed'], padding_idx=0)
        self.embed.weight.requires_grad = False
        self.lstm = nn.LSTM(300, 150, bidirectional=True, num_layers=2, dropout=0.5)

        self.linear = nn.Linear(300, 3)

    def forward(self, data, lengths):
        embeded_tweets = self.embed(data)
        embeded_tweets_packed = pack_padded_sequence(embeded_tweets, lengths, batch_first=True)
        lstm = self.lstm(embeded_tweets_packed)[0]
        outputs = pad_packed_sequence(lstm)[0]
        lin = self.linear(outputs[-1])



        return lin

