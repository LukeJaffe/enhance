#!/usr/bin/env python3

'''Train CIFAR10 with PyTorch.'''
from IPython import embed

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import sys
import argparse

from models import *
from utils import progress_bar
from torch.autograd import Variable

import random
from PIL import Image
import numpy as np
import skimage
import json
import progressbar

import dataset

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--super', action='store_true')
parser.add_argument('--bicubic', action='store_true')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

acc_dict = {"train": [], "test": []}

stats_file = None
stats_file = 'stats2.json'

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Scale(64),
    transforms.RandomCrop(64, padding=8),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.53226813,  0.43759231,  0.34356611), (0.27382627,  0.27767733,  0.28433699))
])

transform_test = transforms.Compose([
    transforms.Scale(64),
    transforms.ToTensor(),
    transforms.Normalize((0.53226813,  0.43759231,  0.34356611), (0.27382627,  0.27767733,  0.28433699))
])


# Load data
trainset = dataset.Food101(root='./data', train=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

testset = dataset.Food101(root='./data', train=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)

#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    net = VGG('VGG19_64')
    #net = VGG('VGG11_48') # 22s train, 2.3s test, 10.3% test after 1 epoch
    #net = VGG('VGG19_64') # 22s train, 2.3s test, 10.3% test after 1 epoch
    #net = ResNet18() # 22s train, 2.5s test, 24.8% test after 1 epoch
    #net = ResNeXt29_2x64d() #50s train, 3.5s test, 40.6% test after 1 epoch
    #net = DenseNet121() #67s train, 5.6s test, 44.9% test after 1 epoch
    #net = GoogLeNet() # 67s train, 5.2s test, 51.5% test after 1 epoch

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return correct/total 

def val(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)

        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    return correct/total


for epoch in range(start_epoch, start_epoch+350):
    train_acc = train(epoch)
    test_acc = val(epoch)
    acc_dict["train"].append(train_acc)
    acc_dict["test"].append(test_acc)
    with open(stats_file, 'w', encoding='utf8') as fp:
        json.dump(acc_dict, fp)
    if epoch == 150:
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    elif epoch == 250: 
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    if epoch%10 == 0:
        state = {
            'net': net.module if use_cuda else net,
            'acc': test_acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc
