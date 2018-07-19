import numpy as np
import pandas as pd
import torch

from torch.utils.data import *
import torch.nn.functional as F
import torch.optim as optim

from rnn_model import RnnModel, load_glove, create_vocab, tokenize_tweets, tokenize_tweets2
from preprocessing import read_files, pad_sequences, ReaderDataset
from params import *





if __name__ == '__main__':

    dataset = read_files(params['data_path'])
    #dataset['tokenized_tweet'] = tokenize_tweets(dataset)

    dataset = dataset.sample(frac=0.2).reset_index()

    dataset_train = dataset[:int(round(len(dataset)*0.8))]
    dataset_train = dataset_train[:int(round(len(dataset_train)*0.8))]
    dataset_valid = dataset_train[int(round(len(dataset_train)*0.8)):]
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
                                                   batch_size=20,
                                                   shuffle=False)

    valid_data = torch.utils.data.DataLoader(dataset=valid,
                                                   batch_size=20,
                                                   shuffle=False)

    test_data = torch.utils.data.DataLoader(dataset=test,
                                                  batch_size=100,
                                                  shuffle=False)



    glove = load_glove(params['embedding_file'])
    tweets = tokenize_tweets(dataset_train)
    vocabulary, dictionary = create_vocab(tweets, glove)
    params['n_embed'] = vocabulary.shape[0]
    model = RnnModel(params=params)
    model.embed.weight.data.copy_(torch.from_numpy(vocabulary))

    labels_dict = {'neutral': 0,
                   'negative': 1,
                   'positive': 2}


    for epoch in range(2):
        print('Epoch {}'.format(epoch))
        training_loss = 0.0

        predicted_values_train = []
        true_values_train = []

        predicted_values_test = []
        true_values_test = []

        for batch_idx, ex in enumerate(train_data):
            tokenized_tweets = tokenize_tweets2(ex['tokenized_tweet'])
            indexed_labels = [labels_dict[i] for i in ex['label']]

            tensor_dictionary = pad_sequences(indexed_labels, tokenized_tweets, dictionary)

            data = tensor_dictionary['tweet_tensor'].cuda()
            label = tensor_dictionary['label_tensor'].cuda()
            model.cuda().train()

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
            tokenized_tweets = tokenize_tweets2(ex['tokenized_tweet'])
            indexed_labels = [labels_dict[i] for i in ex['label']]

            tensor_dictionary = pad_sequences(indexed_labels, tokenized_tweets, dictionary)

            data = tensor_dictionary['tweet_tensor'].cuda()
            label = tensor_dictionary['label_tensor'].cuda()

            model.cuda().eval()

            predictions= model(data, tensor_dictionary['length'])
            indexed_labels = torch.LongTensor([labels_dict[i] for i in ex['label']]).cuda()
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

    for batch_idx, (data, label) in enumerate(test_data):
        tokenized_tweets = tokenize_tweets2(ex['tokenized_tweet'])
        indexed_labels = [labels_dict[i] for i in ex['label']]

        tensor_dictionary = pad_sequences(indexed_labels, tokenized_tweets, dictionary)

        data = tensor_dictionary['tweet_tensor'].cuda()
        label = tensor_dictionary['label_tensor'].cuda()

        model.cuda().eval()

        predictions = model(data)
        predictions = F.log_softmax(predictions, dim=1)
        loss = F.nll_loss(predictions, label)

        validation_loss += loss.data

        preds = list(torch.max(predictions, 1)[1].cpu().numpy())

        predicted_values_test.extend(preds)
        true_values_test.extend(list(label.cpu().numpy()))

        accuracy_test = np.sum(np.array(predicted_values_test) == np.array(true_values_test))

print('acc on test set:' + ' ' + str(accuracy_test / 10000))
