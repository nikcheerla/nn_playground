import torch as th
import torch.nn as nn
import torch.nn.functional as F

from torchsample.modules import ModuleTrainer
from torchsample.callbacks import *
from torchsample.regularizers import *
from torchsample.constraints import *
from torchsample.initializers import *
from torchsample.metrics import *


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

import os
from torchvision import datasets
dataset = datasets.MNIST("../data", train=True, download=True)
x_train, y_train = th.load(os.path.join(dataset.root, 'processed/training.pt'))
x_test, y_test = th.load(os.path.join(dataset.root, 'processed/test.pt'))

x_train = x_train.float()
y_train = y_train.long()
x_test = x_test.float()
y_test = y_test.long()

x_train = x_train / 255.
x_test = x_test / 255.
x_train = x_train.unsqueeze(1)
x_test = x_test.unsqueeze(1)

# only train on a subset
x_train = x_train[:10000]
y_train = y_train[:10000]
x_test = x_test[:1000]
y_test = y_test[:1000]


NUM_LAYERS = 6
MAX_DILATION = 3
FILTERS = 32
KERNEL_SIZE = 3
DROPOUT_RATE = 0


def one_hot_encode_tensor(layer_num=0, dilation_val=1, num_layers=NUM_LAYERS, max_dilation=MAX_DILATION):
    #Constant variable array
    array = np.zeros((1, num_layers+max_dilation))
    array[0, layer_num] = 1
    array[0, (num_layers - 1):(num_layers + dilation_val - 1)] = 1
    array = Variable(torch.Tensor(array).float(), requires_grad=False)

    return array


class ConvolutionNode(nn.Module):
    def __init__(self, kernel_size, dilation_rate, num_inputs=4):
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.layer = nn.Conv2d(1, )


#Net contains a hypernet which parametrizes conv layers
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.input = nn.Conv2d(1, FILTERS, kernel_size=(1, 1), padding=1)

        self.conv1 = nn.Conv2d(FILTERS, FILTERS, kernel_size=KERNEL_SIZE, padding=(1, 1))
        self.conv2 = nn.Conv2d(FILTERS, FILTERS, kernel_size=KERNEL_SIZE, padding=(1, 1))
        self.conv3 = nn.Conv2d(FILTERS, FILTERS, kernel_size=KERNEL_SIZE, padding=(1, 1))
        self.conv4 = nn.Conv2d(FILTERS, FILTERS, kernel_size=KERNEL_SIZE, padding=(1, 1))
        self.conv5 = nn.Conv2d(FILTERS, FILTERS, kernel_size=KERNEL_SIZE, padding=(1, 1))
        self.fc1 = nn.Linear(32, 512)
        self.fc2 = nn.Linear(512, 10)

        if torch.cuda.is_available():
            self.cuda()
        
    def forward(self, x):

        if torch.cuda.is_available():
            x = x.cuda()

        x = F.relu(self.input(x))

        #x = F.relu(F.conv2d(x, self.conv1._parameters['weight'], bias=self.conv1._parameters['bias'], padding=1, dilation=1))
        x = F.dropout(x, p=DROPOUT_RATE, training=self.training)
        x = F.max_pool2d(x, 2, stride=1)
        
        x = F.relu(F.conv2d(x, self.conv2._parameters['weight'], bias=self.conv2._parameters['bias'], padding=2, dilation=2))
        x = F.dropout(x, p=DROPOUT_RATE, training=self.training)
        x = F.max_pool2d(x, 2, stride=1)
        
        x = F.relu(F.conv2d(x, self.conv3._parameters['weight'], bias=self.conv3._parameters['bias'], padding=4, dilation=4)) 
        x = F.dropout(x, p=DROPOUT_RATE, training=self.training)
        x = F.max_pool2d(x, 2, stride=1)
        
        #x = F.relu(F.conv2d(x, self.conv4._parameters['weight'], bias=self.conv4._parameters['bias'], padding=8, dilation=8))
        #x = F.max_pool2d(x, 2, stride=1)

        x = F.relu(F.conv2d(x, self.conv5._parameters['weight'], bias=self.conv5._parameters['bias'], padding=8, dilation=8))
        print (x.size())
        x1 = F.max_pool2d(x, x.size(2)).view(x.size(0), -1)
        x2 = x.mean(2).mean(3).view(x.size(0), -1)
        x = x1 + x2 

        x = F.dropout(x, p=DROPOUT_RATE, training=self.training)
        x = F.log_softmax(self.fc2(F.sigmoid(self.fc1(x))))
        return x

# Define your model EXACTLY as if you were using nn.Module
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2) 
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        #self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(288, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
    	
        if torch.cuda.is_available():
            x = x.cuda()

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.dropout(x, p=DROPOUT_RATE, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=DROPOUT_RATE, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(x.size(0), -1)
        #print (x.size())
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


# Define your model EXACTLY as if you were using nn.Module
class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
    	x = x.cuda()
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


model = Net()
model = model.cuda(0)
trainer = ModuleTrainer(model)

callbacks = [EarlyStopping(patience=10),
             ReduceLROnPlateau(factor=0.5, patience=5)]
regularizers = [L1Regularizer(scale=1e-3, module_filter='conv*'),
                L2Regularizer(scale=1e-5, module_filter='fc*')]
constraints = [UnitNorm(frequency=3, unit='batch', module_filter='fc*'),
               MaxNorm(value=2., lagrangian=True, scale=1e-2, module_filter='conv*')]
initializers = [XavierUniform(bias=False, module_filter='fc*')]
metrics = [CategoricalAccuracy(top_k=1)]

trainer.compile(loss='nll_loss',
                optimizer='adadelta',
                regularizers=regularizers,
                constraints=constraints,
                initializers=initializers,
                metrics=metrics)

summary = trainer.summary([1,28,28])
print(summary)

trainer.fit(x_train, y_train, 
          val_data=(x_test, y_test),
          nb_epoch=40, 
          batch_size=32,
          verbose=1, cuda_device=0)