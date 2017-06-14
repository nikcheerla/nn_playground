
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

X_train, X_test = torch.Tensor(X_train).float().cuda(), torch.Tensor(X_test).float().cuda()
Y_train, Y_test = torch.Tensor(Y_train).long().cuda(), torch.Tensor(Y_test).long().cuda()


def one_hot_encode_tensor(num, max_size=4):
    #Constant variable array
    array = Variable(torch.Tensor(np.zeros((1, max_size))).float(), requires_grad=False)
    array[0, num] = 1
    return array


class HyperNet(nn.Module):
    def __init__(self):
        super(HyperNet, self).__init__()

        self.fc1 = nn.Linear(4, 32)
        self.conv1 = nn.Conv2d(2, 4, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(4, 8, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(8, 8, kernel_size=(3, 3), padding=(1, 1))
        self.conv4 = nn.Conv2d(8, 1, kernel_size=(3, 3), padding=(1, 1))
        
    def forward(self, x):
        x = x.cuda()
        x = F.tanh(self.fc1(x))
        x = x.view(1, 2, 4, 4)
        x = F.dropout(F.relu(self.conv1(x)), p=0.3, training=self.training)
        #print (x.size())
        x = F.upsample_bilinear(x, scale_factor=3)
        x = F.dropout(F.relu(self.conv2(x)), p=0.3, training=self.training)
        x = F.upsample_bilinear(x, scale_factor=2)
        x = F.dropout(F.relu(self.conv3(x)), p=0.3, training=self.training)
        x = F.upsample_bilinear(x, scale_factor=2)
        x = F.tanh(self.conv4(x))
        #print (x.size())
        x = x.view(16, 3, 16, 3)
        x = x.permute(0, 2, 1, 3) * 0.1
        #print ("Weight1 mean: ", self.fc1._parameters['weight'].mean().data.numpy())
        #print ("Weight2 mean: ", self.fc2._parameters['weight'].mean().data.numpy())
        #print ("Activation mean: ", x.mean().data.numpy())
        return x

#Net contains a hypernet which parametrizes conv layers
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hypernet = HyperNet().cuda()
        self.input = nn.Conv2d(1, 16, kernel_size=(1, 1))

        self.conv1 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1))
        self.conv4 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1))
        self.fc = nn.Linear(16*7*7, 10)
        
    def forward(self, x):
        x = x.cuda()
        x = self.input(x)

        self.hypernet.training = self.training
        conv1_weights = self.hypernet(one_hot_encode_tensor(0))
        conv2_weights = self.hypernet(one_hot_encode_tensor(1))
        conv3_weights = self.hypernet(one_hot_encode_tensor(2))
        conv4_weights = self.hypernet(one_hot_encode_tensor(3))

        x = F.relu(F.conv2d(x, conv1_weights, bias=self.conv1._parameters['bias'], padding=1))
        x = F.relu(F.conv2d(x, conv2_weights, bias=self.conv2._parameters['bias'], padding=1))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.max_pool2d(x, 2)
        x = F.relu(F.conv2d(x, conv3_weights, bias=self.conv3._parameters['bias'], padding=1))
        x = F.relu(F.conv2d(x, conv4_weights, bias=self.conv4._parameters['bias'], padding=1))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.fc(x)
        return x

net = Net().cuda()
print(net)

model = ModuleTrainer(net)

print(model)

callbacks = [EarlyStopping(patience=10),
             ReduceLROnPlateau(factor=0.5, patience=5)]
metrics = [CategoricalAccuracy()]

model.compile(loss=F.cross_entropy,
                optimizer='adadelta',
                metrics=metrics)

model.summary([1, 28, 28])

model.fit(X_train, Y_train, batch_size=128, nb_epoch=5, verbose=1, val_data=(X_test, Y_test), cuda_device=0)

#IPython.embed()
