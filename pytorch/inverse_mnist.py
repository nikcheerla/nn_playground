
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
import random, sys, os

from tqdm import tqdm


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(10, 16*7*7),
            nn.ReLU(),
            nn.Tanh(),
        )
        self.vgg1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.vgg2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        
        self.output = nn.Conv2d(16, 1, kernel_size=(1, 1))
        
    def forward(self, x):
        #print ("Data1!!!!", x.data.numpy())
        x = self.fc(x)
        x = x.clamp(-10, 10)
        #print ("Data!!!!", x.data.numpy())
        #if random.randint(0, 20) == 0:
        #    sys.exit()
        x = x.view(-1, 16, 7, 7)
        x = self.vgg1(x)
        x = self.vgg2(x)
        x = self.output(x)
        return x

net = Net()
print(net)

def get_iterator(mode):
    ds = MNIST(root='./', download=True, train=mode)
    data = getattr(ds, 'train_data' if mode else 'test_data')
    labels = getattr(ds, 'train_labels' if mode else 'test_labels')
    labels = torch.FloatTensor(np_utils.to_categorical(labels.numpy()))
    tds = tnt.dataset.TensorDataset([labels[0:4000], data[0:4000, np.newaxis]])
    return tds.parallel(batch_size=4, num_workers=4, shuffle=mode)

optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

engine = Engine()
loss_fn = torch.nn.MSELoss(size_average=False)
meter_loss = tnt.meter.AverageValueMeter()

def h(sample):
    inputs = Variable(torch.FloatTensor(sample[0].float()))
    targets = Variable(torch.FloatTensor(sample[1].float() / 255.0))
    
    if sample[2]:
        net.train()
    else:
        net.eval()
    o = net(inputs)
    #print (o.data.numpy())
    #print (targets.data.numpy())
    return loss_fn(o, targets), o

def reset_meters():
    meter_loss.reset()

def on_sample(state):
    state['sample'].append(state['train'])

def on_forward(state):
    meter_loss.add(state['loss'].data[0])

def on_start_epoch(state):
    reset_meters()
    state['iterator'] = tqdm(state['iterator'])

def on_end_epoch(state):
    print 'Training loss: %.4f' % (meter_loss.value()[0])
    # do validation at the end of each epoch
    reset_meters()
    engine.test(h, get_iterator(False))
    print 'Testing loss: %.4f' % (meter_loss.value()[0])

engine.hooks['on_sample'] = on_sample
engine.hooks['on_forward'] = on_forward
engine.hooks['on_start_epoch'] = on_start_epoch
engine.hooks['on_end_epoch'] = on_end_epoch
engine.train(h, get_iterator(True), maxepoch=10, optimizer=optimizer)

