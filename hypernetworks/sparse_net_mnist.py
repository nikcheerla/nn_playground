
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

X_train, X_test = torch.Tensor(X_train).float(), torch.Tensor(X_test).float()
Y_train, Y_test = torch.Tensor(Y_train).long(), torch.Tensor(Y_test).long()



NUM_LAYERS = 6
MAX_DILATION = 3
FILTERS = 32
KERNEL_SIZE = 3
DROPOUT_RATE = 0.4





class ConvolutionNode():
     def __init__(self, dilation_rate=1, kernel_size=3, activation=F.relu):
        super(ConvolutionNode, self).__init__()
        self.activation = activation
        self.dilation_rate = dilation_rate
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv2d(1, 1, kernel_size=kernel_size, dilation_rate=dilation_rate)
        self.cache_tensor = None
        
    def forward(self, x):

        if torch.cuda.is_available():
            x = x.cuda()

        batch_size, width, height = x.size(0), x.size(1), x.size(2)
        x = x.view(batch_size, 1, width, height)
        x = self.activation(self.conv1(x))
        width_new, height_new = x.size(2), x.size(3)
        xd, yd = width - width_new, height - height_new
        x = F.pad(x, (xd//2, xd - xd//2, yd//2, yd - yd//2), mode='reflect')
        x = x.view(batch_size, width, height)
        self.cache_tensor = x

        return x

    @classmethod
    def key(cls, node):
        return (node.dilation_rate, node.kernel_size)



class HyperNet(nn.Module):
    def __init__(self):
        super(HyperNet, self).__init__()

        self.sparse_nodes = []
        for i in range(0, 200):
            dilation_rate = random.choice([1, 2, 4, 6], probs=[0.25, 0.3, 0.3, 0.15])
            kernel_size = random.choice([3, 5], probs=[0.75, 0.25])
            self.sparse_nodes.append(ConvolutionNode(dilation_rate=dilation_rate, kernel_size=kernel_size))
        self.sparse_nodes = sorted(self.sparse_nodes, key=ConvolutionNode.key)
        self.sparse_nodes = nn.ModuleList(self.sparse_nodes)
        self.N = len(self.sparse_nodes)

        self.connectivity_matrix = np.zeros((self.N, self.N))
        self.mask = np.zeros((self.N, self.N))

        for i in range(0, self.N):
            for j in range(0, i-1):
                self.mask[i, j] = 1
                self.connectivity_matrix[i, j] = random.random()

        self.fc1 = nn.Linear(self.N, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):

        if torch.cuda.is_available():
            x = x.cuda()

        x = F.tanh(self.fc1(x))
        x = x.view(1, 2, FILTERS/8, FILTERS/8)
        x = F.dropout(F.relu(self.conv1(x)), p=DROPOUT_RATE, training=self.training)
        x = F.upsample_bilinear(x, scale_factor=2)
        x = F.dropout(F.relu(self.conv2(x)), p=DROPOUT_RATE, training=self.training)
        x = F.upsample_bilinear(x, scale_factor=2)
        x = F.dropout(F.relu(self.conv3(x)), p=DROPOUT_RATE, training=self.training)
        x = F.upsample_bilinear(x, scale_factor=2)
        x = F.tanh(self.conv4(x))
        #print (x.size())
        x = x.view(FILTERS, FILTERS, KERNEL_SIZE, KERNEL_SIZE)
        x = x.permute(0, 1, 2, 3) * 0.8
        #print ("Weight1 mean: ", self.fc1._parameters['weight'].mean().data.numpy())
        #print ("Weight2 mean: ", self.fc2._parameters['weight'].mean().data.numpy())
        #print ("Activation mean: ", x.mean().data.numpy())
        return x

#Net contains a hypernet which parametrizes conv layers
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hypernet = HyperNet()

        self.input = nn.Conv2d(1, FILTERS, kernel_size=(1, 1))

        self.conv1 = nn.Conv2d(FILTERS, FILTERS, kernel_size=KERNEL_SIZE, padding=(1, 1))
        self.conv2 = nn.Conv2d(FILTERS, FILTERS, kernel_size=KERNEL_SIZE, padding=(1, 1))
        self.conv3 = nn.Conv2d(FILTERS, FILTERS, kernel_size=KERNEL_SIZE, padding=(1, 1))
        self.conv4 = nn.Conv2d(FILTERS, FILTERS, kernel_size=KERNEL_SIZE, padding=(1, 1))
        self.conv5 = nn.Conv2d(FILTERS, FILTERS, kernel_size=KERNEL_SIZE, padding=(1, 1))
        self.conv6 = nn.Conv2d(FILTERS, FILTERS, kernel_size=KERNEL_SIZE, padding=(1, 1))
        self.fc1 = nn.Linear(32, 512)
        self.fc2 = nn.Linear(512, 10)

        if torch.cuda.is_available():
            self.hypernet.cuda()
            self.cuda()
        
    def forward(self, x):

        if torch.cuda.is_available():
            x = x.cuda()

        x = self.input(x)

        self.hypernet.training = self.training
        conv1_weights = self.hypernet(one_hot_encode_tensor(layer_num=0, dilation_val=1))
        conv2_weights = self.hypernet(one_hot_encode_tensor(layer_num=1, dilation_val=1))
        conv3_weights = self.hypernet(one_hot_encode_tensor(layer_num=2, dilation_val=2))
        conv4_weights = self.hypernet(one_hot_encode_tensor(layer_num=3, dilation_val=2))
        conv5_weights = self.hypernet(one_hot_encode_tensor(layer_num=4, dilation_val=3))
        conv6_weights = self.hypernet(one_hot_encode_tensor(layer_num=5, dilation_val=3))

        x = F.relu(F.conv2d(x, self.conv1._parameters['weight'], bias=self.conv1._parameters['bias'], padding=1, dilation=1))
        x = F.relu(F.conv2d(x, self.conv2._parameters['weight'], bias=self.conv2._parameters['bias'], padding=1, dilation=1))
        x = F.dropout(x, p=DROPOUT_RATE, training=self.training)
        x = F.max_pool2d(x, 2, stride=1)

        x = F.relu(F.conv2d(x, self.conv3._parameters['weight'], bias=self.conv3._parameters['bias'], padding=1, dilation=2))
        x = F.relu(F.conv2d(x, self.conv4._parameters['weight'], bias=self.conv4._parameters['bias'], padding=1, dilation=2))
        x = F.dropout(x, p=DROPOUT_RATE, training=self.training)
        #x = F.max_pool2d(x, 2, stride=1)
        #x = F.relu(F.conv2d(x, self.conv5._parameters['weight'], bias=self.conv5._parameters['bias'], padding=1, dilation=4))
        #x = F.relu(F.conv2d(x, self.conv6._parameters['weight'], bias=self.conv6._parameters['bias'], padding=1, dilation=4))
        x = F.max_pool2d(x, x.size(2), stride=1)

        x = x.view(x.size(0), -1)
        x = F.dropout(x, p=DROPOUT_RATE, training=self.training)
        x = F.softmax(self.fc2(F.sigmoid(self.fc1(x))))
        return x

net = Net()
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

model.fit(X_train, Y_train, batch_size=8, nb_epoch=4, verbose=1, val_data=(X_test, Y_test), cuda_device=0)

IPython.embed()
