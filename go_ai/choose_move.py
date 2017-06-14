import subprocess
import sys

import cPickle as pickle


import numpy as np
import theano

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

BOARD_SIZE = 5

file1 = sys.argv[1]
filename = sys.argv[2]

if "9x9" in file1:
    BOARD_SIZE = 9
if "19x19" in file1:
    BOARD_SIZE = 19


with open(filename, 'rb') as f:
	net = pickle.load(f)


board_path = file1
#print board_path
board = open(board_path, "rb")
board_arr = np.zeros((28, 28))
board_arr = board_arr.astype(int)

turn, won  = map(int, board.readline().split())
#1 is always about to play
for i in range(0, BOARD_SIZE):
    line = board.readline();
    for j in range(0, BOARD_SIZE):
        if int(line[j]) == 0:
            board_arr[i][j] = 0
            continue
        if turn == 1:
            board_arr[i][j] = 3 - int(line[j])
        elif turn == 2:
            board_arr[i][j] = int(line[j])

#print board_arr

maxi = 0
maxj = 0
maxprob = -1


for i in range(0, BOARD_SIZE):
    for j in range(0, BOARD_SIZE):
        if int(board_arr[i][j]) == 0:
            board_arr[i][j] = 2
            prob = net.predict(board_arr.reshape((-1, 1, 28, 28)))[0]
            #print prob
            if prob >= maxprob:
                maxi = i
                maxj = j
                maxprob = prob
            board_arr[i][j] = 0

board_arr[maxi][maxj] = 2

turn = 3 - turn;
#now its the other person's turn

print "\n\nplaced stone at: " + `maxi` + ", " + `maxj`
print "prob: " + `maxprob` + "\n"

outfile = open(file1, 'wb')

st = `turn` + " " + `won` + "\n"
for i in range(0, BOARD_SIZE):
    for j in range(0, BOARD_SIZE):
        if int(board_arr[i][j]) == 0:
            st += `0`
        elif 1 == board_arr[i][j]:
            st += `turn`
        else:
            st += `3 - turn`
    st += "\n"

print st
outfile.write(st)
outfile.close()



