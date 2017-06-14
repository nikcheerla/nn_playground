
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from keras.datasets import mnist
from keras.utils import np_utils
from torch.utils.data import TensorDataset, DataLoader
import IPython

import torch.optim
import torchnet as tnt
from torchvision.datasets.mnist import MNIST
from torchnet.engine import Engine
from torch.nn.init import kaiming_normal
import random

from tqdm import tqdm


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.input = nn.Conv2d(1, 16, kernel_size=(1, 1))
        self.vgg1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)))
        self.vgg2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)))
        self.vgg3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)))
        self.fc = nn.Linear(16*7*7, 10)
        
    def forward(self, x):
        x = self.input(x)
        f1 = random.choice([self.vgg1, self.vgg2, self.vgg3])
        f2 = random.choice([self.vgg1, self.vgg2, self.vgg3])
        x = f1(f2((x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

net = Net()
print(net)

def get_iterator(mode):
    ds = MNIST(root='./', download=True, train=mode)
    data = getattr(ds, 'train_data' if mode else 'test_data')
    labels = getattr(ds, 'train_labels' if mode else 'test_labels')
    tds = tnt.dataset.TensorDataset([data[:, np.newaxis], labels])
    return tds.parallel(batch_size=256, num_workers=4, shuffle=mode)

optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

engine = Engine()
meter_loss = tnt.meter.AverageValueMeter()
classerr = tnt.meter.ClassErrorMeter(accuracy=True)

def h(sample):
    inputs = Variable(sample[0].float() / 255.0)
    targets = Variable(torch.LongTensor(sample[1]))
    if sample[2]:
        net.train()
    o = net(inputs)
    return F.cross_entropy(o, targets), o

def reset_meters():
    classerr.reset()
    meter_loss.reset()

def on_sample(state):
    state['sample'].append(state['train'])

def on_forward(state):
    classerr.add(state['output'].data, torch.LongTensor(state['sample'][1]))
    meter_loss.add(state['loss'].data[0])

def on_start_epoch(state):
    reset_meters()
    state['iterator'] = tqdm(state['iterator'], ncols=80)

def on_end_epoch(state):
    print 'Training loss: %.4f, accuracy: %.2f%%' % (meter_loss.value()[0], classerr.value()[0])
    # do validation at the end of each epoch
    reset_meters()
    engine.test(h, get_iterator(False))
    print 'Testing loss: %.4f, accuracy: %.2f%%' % (meter_loss.value()[0], classerr.value()[0])

engine.hooks['on_sample'] = on_sample
engine.hooks['on_forward'] = on_forward
engine.hooks['on_start_epoch'] = on_start_epoch
engine.hooks['on_end_epoch'] = on_end_epoch
engine.train(h, get_iterator(True), maxepoch=10, optimizer=optimizer)

