
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from keras.datasets import mnist, cifar10
from keras.utils import np_utils
from torch.utils.data import TensorDataset, DataLoader
from progressbar import ProgressBar
import IPython

import torch.optim
from torchvision.datasets.mnist import MNIST
from torch.nn.init import kaiming_normal

from torchsample.modules import ModuleTrainer
from torchsample.callbacks import *
from torchsample.regularizers import *
from torchsample.constraints import *
from torchsample.initializers import *
from torchsample.metrics import *
import random, sys

batch_size = 32
num_classes = 10
epochs = 200
data_augmentation = True

# The data, shuffled and split between train and test sets:
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
print('x_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

X_train = X_train.astype('float32')/255.0
X_test = X_test.astype('float32')/255.0

Y_train, Y_test = 1.0*Y_train, 1.0*Y_test
X_train, X_test = torch.Tensor(X_train).float().cuda(), torch.Tensor(X_test).float().cuda()
Y_train, Y_test = torch.Tensor(Y_train).long().cuda(), torch.Tensor(Y_test).long().cuda()


NUM_LAYERS = 6
MAX_DILATION = 3
FILTERS = 32
KERNEL_SIZE = 3
DROPOUT_RATE = 0.4

#Net contains a hypernet which parametrizes conv layers
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.hypernet = HyperNet().cuda()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv5 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv6 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv7 = nn.Conv2d(64, 32, kernel_size=(1, 1))
        self.fc = nn.Linear(32, 256)
        self.out = nn.Linear(256, 10)
        
    def forward(self, x):
        x = x.cuda()
        print (x.size())

        IPython.embed()

        #self.hypernet.training = self.training
        #conv1_weights = self.hypernet(one_hot_encode_tensor(0))
        #conv2_weights = self.hypernet(one_hot_encode_tensor(1))
        #conv3_weights = self.hypernet(one_hot_encode_tensor(2))
        #conv4_weights = self.hypernet(one_hot_encode_tensor(3))

        x = F.relu(self.conv1(x))
        x = F.relu(F.conv2d(x, self.conv2._parameters['weight'], bias=self.conv2._parameters['bias'], padding=1, dilation=1))
        x = F.dropout(x, p=DROPOUT_RATE, training=self.training)
        x = F.max_pool2d(x, 2, stride=1)

        x = F.relu(F.conv2d(x, self.conv3._parameters['weight'], bias=self.conv3._parameters['bias'], padding=1, dilation=2))
        x = F.relu(F.conv2d(x, self.conv4._parameters['weight'], bias=self.conv4._parameters['bias'], padding=1, dilation=2))
        x = F.dropout(x, p=DROPOUT_RATE, training=self.training)
        x = F.max_pool2d(x, 2, stride=1)

        x = F.relu(F.conv2d(x, self.conv5._parameters['weight'], bias=self.conv5._parameters['bias'], padding=1, dilation=4))
        x = F.relu(F.conv2d(x, self.conv6._parameters['weight'], bias=self.conv6._parameters['bias'], padding=1, dilation=4))
        x = F.max_pool2d(x, 2, stride=1)
        x = F.tanh(F.conv2d(x, self.conv7._parameters['weight'], bias=self.conv7._parameters['bias'], padding=1, dilation=1))

        x = F.max_pool2d(x, x.size(2))
        x = x.view(x.size(0), -1)

        x = F.sigmoid(self.fc(x))
        x = F.softmax(self.out(x))

        return x

net = Net().cuda()
print(net)

model = ModuleTrainer(net)

print(model)

callbacks = [EarlyStopping(patience=10),
             ReduceLROnPlateau(factor=0.5, patience=5)]
metrics = [CategoricalAccuracy()]

model.compile(loss=F.cross_entropy,
                optimizer='rmsprop',
                metrics=metrics)

#model.summary([1, 32, 32])

model.fit(X_train, Y_train, batch_size=1, nb_epoch=20, verbose=1, val_data=(X_test, Y_test), cuda_device=0)

#IPython.embed()
