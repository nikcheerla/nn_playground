
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from keras.datasets import mnist
from keras.utils import np_utils
from torch.utils.data import TensorDataset, DataLoader
from progressbar import ProgressBar
import IPython

import torch.optim
import torchnet as tnt
from torchvision.datasets.mnist import MNIST
from torchnet.engine import Engine
from torch.nn.init import kaiming_normal

from torchsample.modules import ModuleTrainer
from torchsample.callbacks import *
from torchsample.regularizers import *
from torchsample.constraints import *
from torchsample.initializers import *
from torchsample.metrics import *
import random, sys

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train = X_train.astype('float32')/255.0
X_test = X_test.astype('float32')/255.0

Y_train, Y_test = Y_train*1.0, Y_test*1.0

print ('X_train shape:', X_train.shape)
print ('Y_train shape:', Y_train.shape)
print (X_train.shape[0], 'train samples')
print (X_test.shape[0], 'test samples')

X_train, X_test = torch.Tensor(X_train).float(), torch.Tensor(X_test).float()
Y_train, Y_test = torch.Tensor(Y_train).long(), torch.Tensor(Y_test).long()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.input = nn.Conv2d(1, 16, kernel_size=(1, 1))
        self.conv1 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1))
        self.conv4 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1))
        self.fc = nn.Linear(16*7*7, 10)
        
    def forward(self, x):
        x = self.input(x)
        x = F.relu(F.conv2d(x, self.conv1._parameters['weight'], bias=self.conv1._parameters['bias'], padding=1))
        x = F.relu(F.conv2d(x, self.conv2._parameters['weight'], bias=self.conv2._parameters['bias'], padding=1))
        x = F.max_pool2d(x, 2)
        x = F.relu(F.conv2d(x, self.conv3._parameters['weight'], bias=self.conv3._parameters['bias'], padding=1))
        x = F.relu(F.conv2d(x, self.conv4._parameters['weight'], bias=self.conv4._parameters['bias'], padding=1))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

net = Net()
print(net)
model = ModuleTrainer(net)
print(model)

callbacks = [EarlyStopping(patience=10),
             ReduceLROnPlateau(factor=0.5, patience=5)]
metrics = [CategoricalAccuracy(top_k=3)]

model.compile(loss=F.cross_entropy,
                optimizer=torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005),
                metrics=metrics)

model.summary([1, 28, 28])

model.fit(X_train, Y_train, batch_size=128, nb_epoch=5, verbose=1, val_data=(X_test, Y_test))
