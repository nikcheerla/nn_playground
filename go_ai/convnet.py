import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys


from urllib import urlretrieve
import cPickle as pickle
import os
import gzip

import numpy as np
import random
import theano

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

BOARD_SIZE = 9
if len(sys.argv) > 1:
    BOARD_SIZE = int(sys.argv[1])

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_dataset():

    path = './data/' + `BOARD_SIZE` + 'x' + `BOARD_SIZE` + '/'
    size = 0
    for f in os.listdir(path):
        if f.startswith('.'):
            continue
        size = size + 1

    
    X_train = np.zeros((size, 28, 28))
    X_train = X_train.astype(int)
    size = 0
    wins = {};
    count = {};
    for f in os.listdir(path):
        if f.startswith('.'):
            continue
        board_path = os.path.join(path, f)
        board = open(board_path, "rb")
        if random.random() <= 0.004:
            print board_path

        #print board_path

        turn, won  = map(int, board.readline().split())
        #1 is always about to play
        for i in range(0, BOARD_SIZE):
            line = board.readline();
            for j in range(0, BOARD_SIZE):
                if int(line[j]) == 0:
                    X_train[size][i][j] = 0
                    continue
                if turn == 1:
                    X_train[size][i][j] = 3- 2*int(line[j])
                elif turn == 2:
                    X_train[size][i][j] = 2*int(line[j]) - 3
        #X_train[size] = board_arr
        if str(X_train[size]) not in count:
            count[str(X_train[size])] = 0
            wins[str(X_train[size])] = 0
        count[str(X_train[size])] = count[str(X_train[size])] + 1
        if turn == won:
            wins[str(X_train[size])] = wins[str(X_train[size])] + 1
        size = size + 1
    
    X_train1 = np.zeros((len(count), 28, 28))
    X_train1 = X_train1.astype(int)
    y_train = [0.0] * len(count);
    j = 0
    for i in range(0, size):
        if str(X_train[i]) not in count:
            continue
        y_train[j] = int(100*float(wins[str(X_train[i])])/count[str(X_train[i])])
        
        del count[str(X_train[i])]
        del wins[str(X_train[i])]
        X_train1[j] = X_train[i]
        j = j + 1

    X_train = X_train1

    y_train = np.array(y_train);
    y_train = y_train.astype(np.uint8)

    X_train = X_train.reshape((-1, 1, 28, 28))
    return X_train, y_train, size

def train_net():
    X_train, y_train, size = load_dataset()
    print X_train
    print y_train
    net1 = NeuralNet(
        layers=[('input', layers.InputLayer),
                ('conv2d1', layers.Conv2DLayer),
                ('maxpool1', layers.MaxPool2DLayer),
                ('conv2d2', layers.Conv2DLayer),
                ('maxpool2', layers.MaxPool2DLayer),
                ('dropout1', layers.DropoutLayer),
                ('dense', layers.DenseLayer),
                ('dropout2', layers.DropoutLayer),
                ('output', layers.DenseLayer),
                ],
        # input layer
        input_shape=(None, 1, 28, 28),
        # layer conv2d1
        conv2d1_num_filters=32,
        conv2d1_filter_size=(5, 5),
        conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
        conv2d1_W=lasagne.init.GlorotUniform(),  
        # layer maxpool1
        maxpool1_pool_size=(2, 2),    
        # layer conv2d2
        conv2d2_num_filters=32,
        conv2d2_filter_size=(5, 5),
        conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
        # layer maxpool2
        maxpool2_pool_size=(2, 2),
        # dropout1
        dropout1_p=0.5,    
        # dense
        dense_num_units=256,
        dense_nonlinearity=lasagne.nonlinearities.rectify,    
        # dropout2
        dropout2_p=0.5,    
        # output
        output_nonlinearity=lasagne.nonlinearities.softmax,
        output_num_units=101,
        # optimization method params
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,
        max_epochs=4,
        verbose=1,
        )

    # Train the network
    nn = net1.fit(X_train, y_train)

    # Train the network
    nn = net1.fit(X_train, y_train)

    save_object(nn, 'nets/nnetGO' + `BOARD_SIZE` + "x" + `BOARD_SIZE` + "-" + `size`+ '.pkl')


    preds = net1.predict(X_train)
    print preds

#train_net()

