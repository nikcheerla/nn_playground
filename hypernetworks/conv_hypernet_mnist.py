
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
from torchvision.datasets.mnist import MNIST
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

X_train, X_test = torch.Tensor(X_train).float().cuda(), torch.Tensor(X_test).float().cuda()
Y_train, Y_test = torch.Tensor(Y_train).long().cuda(), torch.Tensor(Y_test).long().cuda()


def one_hot_encode_tensor(layer_num=0, dilation_val=1, num_layers=6, max_dilation=3):
    #Constant variable array
    np.zeros((1, num_layers+max_dilation))
    array[0, layer_num] = 1
    array[0, (num_layers - 1):(num_layers + dilation_val - 1)] = 1
    array = Variable(torch.Tensor(array).float(), requires_grad=False)

    return array

filters = 32
kernel_size = (3, 3)
dropout_rate = 0.3

class HyperNet(nn.Module):
    def __init__(self):
        super(HyperNet, self).__init__()

        self.fc1 = nn.Linear(4, 32)
        self.conv1 = nn.Conv2d(2, 4, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(4, 6, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(6, 8, kernel_size=(3, 3), padding=(1, 1))
        self.conv4 = nn.Conv2d(8, 9, kernel_size=(3, 3), padding=(1, 1))
        
    def forward(self, x):
        x = x.cuda()
        x = F.tanh(self.fc1(x))
        x = x.view(1, 2, 4, 4)
        x = F.dropout(F.relu(self.conv1(x)), p=0.3, training=self.training)
        #print (x.size())
        x = F.upsample_bilinear(x, scale_factor=2)
        x = F.dropout(F.relu(self.conv2(x)), p=0.3, training=self.training)
        x = F.upsample_bilinear(x, scale_factor=2)
        x = F.dropout(F.relu(self.conv3(x)), p=0.3, training=self.training)
        x = F.upsample_bilinear(x, scale_factor=2)
        x = F.tanh(self.conv4(x))
        #print (x.size())
        x = x.view(16, 3, 16, )
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
        self.input = nn.Conv2d(1, 32, kernel_size=(1, 1))

        self.conv1 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.fc = nn.Linear(32, 10)
        
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
        x = F.relu(F.conv2d(x, conv4_weights, bias=None, padding=1))
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
