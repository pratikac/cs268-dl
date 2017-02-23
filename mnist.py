from __future__ import print_function
import argparse, math, random
import numpy as np
import os, pdb, sys, json, subprocess, argparse

import torch as th
import torch.nn as nn
from torch.autograd import Variable

import torch.optim as optim
from torchvision import datasets, transforms

p = argparse.ArgumentParser('')
p.add_argument('--lr', type=float, default=0.1, help='Learning rate')
p.add_argument('-b', type=int, default=128, help='Batch size')
p.add_argument('-B', type=int, default=50, help='Epochs')
p.add_argument('-g', type=int, default=-1, help='GPU idx. (-1 for CPU)')
opt = vars(p.parse_args())
print(opt)

if opt['g'] > -1:
    print('Using GPU: ', opt['g'])
    th.cuda.set_device(opt['g'])
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
else:
    print('Using CPU')

kwargs = {}
if opt['g'] > -1:
    kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader = th.utils.data.DataLoader(
    datasets.MNIST('/local2/pratikac/mnist', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=opt['b'], shuffle=True, **kwargs)
test_loader = th.utils.data.DataLoader(
    datasets.MNIST('/local2/pratikac/mnist', train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=opt['b'], shuffle=False, **kwargs)

class View(nn.Module):
    def __init__(self,o):
        super(View, self).__init__()
        self.o = o
    def forward(self,x):
        return x.view(-1, self.o)

class lenet(nn.Module):
    def __init__(self, opt):
        super(lenet, self).__init__()
        self.name = 'lenet'
        opt['d'] = 0.25

        def convbn(ci,co,ksz,psz,p):
            return nn.Sequential(
                nn.Conv2d(ci,co,ksz),
                nn.ReLU(True),
                nn.MaxPool2d(psz,stride=psz),
                nn.BatchNorm2d(co),
                nn.Dropout(p))

        self.m = nn.Sequential(
            convbn(1,20,5,3,opt['d']),
            convbn(20,50,5,2,opt['d']),
            View(50*2*2),
            nn.Linear(50*2*2, 500),
            nn.ReLU(True),
            nn.Dropout(opt['d']),
            nn.Linear(500,10))

    def forward(self, x):
        return self.m(x)

model = lenet(opt)
criterion = nn.CrossEntropyLoss()
if opt['g'] > -1:
    model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model.parameters(), lr=opt['lr'], momentum=0.9)

print('[Start training]')

def lr_schedule(e):
    step, factor = 15, 0.5
    lr = opt['lr']*factor**(e // step)
    print('[LR]: ', lr)

    # pytorch is weird, we have to modify the learning rate
    # for optim.SGD like this
    for g in optimizer.param_groups:
        g['lr'] = lr

def train(e):
    model.train()

    lr_schedule(e)

    maxb = len(train_loader)
    fs, top1 = [], []
    for i, (x,y) in enumerate(train_loader):
        if opt['g'] > -1:
            x, y = x.cuda(), y.cuda()
        x, yvar  = Variable(x), Variable(y)
        bsz = x.size(0)

        optimizer.zero_grad()
        yh = model(x)
        f = criterion(yh, yvar)
        f.backward()

        optimizer.step()

        # get the index of the max log-probability
        pred = yh.data.max(1)[1]
        acc = pred.eq(y).cpu().sum()/float(bsz)
        err = 100.*(1-acc)

        fs.append(f.data[0])
        top1.append(err)

        if i % 100 == 0:
            print('[%2d][%4d/%4d] %2.4f %2.3f%%'%(e, i, maxb, np.mean(fs), np.mean(top1)))
    print('Train: [%2d] %2.4f %2.3f%%'%(e, np.mean(fs), np.mean(top1)))
    print('')

def test(e):
    model.eval()

    maxb = len(test_loader)
    fs, top1 = [], []
    for i, (x,y) in enumerate(test_loader):
        if opt['g'] > -1:
            x, y = x.cuda(), y.cuda()
        x, yvar  = Variable(x, volatile=True), Variable(y, volatile=True)
        bsz = x.size(0)

        yh = model(x)
        f = criterion(yh, yvar)

        # get the index of the max log-probability
        pred = yh.data.max(1)[1]
        acc = pred.eq(y).cpu().sum()/float(bsz)
        err = 100.*(1-acc)

        fs.append(f.data[0])
        top1.append(err)

        if i % 100 == 0:
            print('[%2d][%4d/%4d] %2.4f %2.3f%%'%(e, i, maxb, np.mean(fs), np.mean(top1)))
    print('Test: [%2d] %2.4f %2.3f%%'%(e, np.mean(fs), np.mean(top1)))
    print('')

for e in xrange(opt['B']):
    train(e)
    if e % 5 == 0:
        test(e)
