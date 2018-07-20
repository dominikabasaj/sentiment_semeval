import numpy as np
import pandas as pd
import torch

from torch.utils.data import *
import torch.nn.functional as F
import torch.optim as optim
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons


from rnn_model import RnnModel, load_glove, create_vocab, tokenize_tweets, tokenize_tweets2
from preprocessing import read_files, pad_sequences, ReaderDataset
from params import *





if __name__ == '__main__':

    text_preprocessor = TextPreProcessor(
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

    dataset = read_files(params['data_path'])
    #dataset['tokenized_tweet'] = tokenize_tweets(dataset)
    dataset = dataset[(dataset['category']=='positive') | (dataset['category']=='negative') | (dataset['category']=='neutral')]

    dataset = dataset.sample(frac=1.0).reset_index()

    dataset_train = dataset[:int(round(len(dataset)*0.8))]
    dataset_train = dataset_train[:int(round(len(dataset_train)*0.9))]
    dataset_valid = dataset_train[int(round(len(dataset_train)*0.9)):]
    dataset_test = dataset[int(round(len(dataset)*0.8)):]

    dataset_train.reset_index(inplace=True)
    print(len(dataset_train))
    dataset_valid.reset_index(inplace=True)
    print(len(dataset_valid))
    dataset_test.reset_index(inplace=True)

    train = ReaderDataset(dataset_train)
    valid = ReaderDataset(dataset_valid)
    test = ReaderDataset(dataset_test)

    train_data = torch.utils.data.DataLoader(dataset=train,
                                                   batch_size=100,
                                                   shuffle=False)

    valid_data = torch.utils.data.DataLoader(dataset=valid,
                                                   batch_size=100,
                                                   shuffle=False)

    test_data = torch.utils.data.DataLoader(dataset=test,
                                                  batch_size=100,
                                                  shuffle=False)



    glove = load_glove(params['embedding_file'])
    tweets = tokenize_tweets(dataset_train, text_preprocessor)
    vocabulary, dictionary = create_vocab(tweets, glove)
    params['n_embed'] = vocabulary.shape[0]
    model = RnnModel(params=params)
    model.embed.weight.data.copy_(torch.from_numpy(vocabulary))

    labels_dict = {'neutral': 0,
                   'negative': 1,
                   'positive': 2}


    for epoch in range(5):
        print('Epoch {}'.format(epoch))
        training_loss = 0.0

        predicted_values_train = []
        true_values_train = []

        predicted_values_test = []
        true_values_test = []

        for batch_idx, ex in enumerate(train_data):
            tokenized_tweets = tokenize_tweets2(ex['tokenized_tweet'], text_preprocessor)

            indexed_labels = [labels_dict[i] for i in ex['label']]

            tensor_dictionary = pad_sequences(indexed_labels, tokenized_tweets, dictionary)

            data = tensor_dictionary['tweet_tensor']
            label = tensor_dictionary['label_tensor']
            model.train()

            predictions = model(data, tensor_dictionary['length'])
            predictions = F.log_softmax(predictions, dim=1)

            #indexed_labels = torch.LongTensor([labels_dict[i] for i in ex['label']]).cuda()
            loss = F.nll_loss(predictions, label)

            training_loss += loss.data
            # optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=0.001,  momentum=0.9, nesterov = True)
            optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=0.001)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = list(torch.max(predictions, 1)[1].cpu().numpy())

            predicted_values_train.extend(preds)
            true_values_train.extend(list(label.cpu().numpy()))

            accuracy_train = np.sum(np.array(predicted_values_train) == np.array(true_values_train))

        validation_loss = 0.0
        for batch_idx, ex in enumerate(valid_data):
            tokenized_tweets = tokenize_tweets2(ex['tokenized_tweet'], text_preprocessor)
            indexed_labels = [labels_dict[i] for i in ex['label']]

            tensor_dictionary = pad_sequences(indexed_labels, tokenized_tweets, dictionary)

            data = tensor_dictionary['tweet_tensor']
            label = tensor_dictionary['label_tensor']

            model.eval()

            predictions= model(data, tensor_dictionary['length'])

            predictions = F.log_softmax(predictions, dim=1)
            loss = F.nll_loss(predictions, label)

            validation_loss += loss.data

            preds = list(torch.max(predictions, 1)[1].cpu().numpy())

            predicted_values_test.extend(preds)
            true_values_test.extend(list(label.cpu().numpy()))

            accuracy_test = np.sum(np.array(predicted_values_test) == np.array(true_values_test))

        print("Epoch {0} "
              "|||train_loss: {1:.3f}"
              "|||val_loss: {2:.3f}"
              "|||train acc: {3:.3f}"
              "|||valid acc: {4:.3f}".format(
            epoch,
            training_loss.item() / len(train_data),
            validation_loss.item() / len(valid_data),
            accuracy_train ,
            accuracy_test ))

    predicted_values_test = []
    true_values_test = []

    for batch_idx, ex in enumerate(test_data):
        tokenized_tweets = tokenize_tweets2(ex['tokenized_tweet'], text_preprocessor)
        indexed_labels = [labels_dict[i] for i in ex['label']]

        tensor_dictionary = pad_sequences(indexed_labels, tokenized_tweets, dictionary)

        data = tensor_dictionary['tweet_tensor'].cuda()
        label = tensor_dictionary['label_tensor'].cuda()

        model.cuda().eval()

        predictions = model(data, tensor_dictionary['length'])
        indexed_labels = torch.LongTensor([labels_dict[i] for i in ex['label']]).cuda()
        predictions = F.log_softmax(predictions, dim=1)
        loss = F.nll_loss(predictions, label)

        validation_loss += loss.data

        preds = list(torch.max(predictions, 1)[1].cpu().numpy())

        predicted_values_test.extend(preds)
        true_values_test.extend(list(label.cpu().numpy()))
print(len(dataset_test))
print('acc on test set:' + ' ' + str(accuracy_test))
