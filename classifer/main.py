#!/usr/bin/env python3

'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from torch.autograd import Variable

import random
from PIL import Image
import numpy as np
import skimage
import json

import noise
import upsample

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', '-m', type=str, default='model_epoch_100.pth')
parser.add_argument('--super', action='store_true')
parser.add_argument('--bicubic', action='store_true')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
model = torch.load(args.model)
model = model.cpu()

class GaussianNoise(object):
    def __init__(self, mean=0, var=1.0):
        self.mean = mean
        self.var = var
    def __call__(self, img):
        arr = np.array(img)
        noised = (skimage.util.random_noise(arr.astype(float)/255.0, mode='gaussian', mean=self.mean, var=self.var)*255.0).astype(np.uint8)
        return Image.fromarray(noised)

class SuperResolve(object):
    def __init__(self, model):
        self.model = model
    def __call__(self, img):
        ybr = img.convert('YCbCr')
        y, cb, cr = ybr.split()
        input = torch.autograd.Variable(torchvision.transforms.ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
        #input = input.cuda()
        out = self.model(input)
        out = out.cpu()
        out_img_y = out.data[0].numpy()
        out_img_y *= 255.0
        out_img_y = out_img_y.clip(0, 255)
        out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

        out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
        out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
        out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
        return out_img

acc_dict = {"train": [], "test": []}

stats_file = None
if args.super:
    stats_file = 'super_stats.json'
elif args.bicubic:
    stats_file = 'bicubic_stats.json'
else:
    stats_file = 'regular_stats.json'

# Data
print('==> Preparing data..')
if args.super:
    transform_train = transforms.Compose([
        #transforms.RandomCrop(96, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
elif args.bicubic:
    transform_train = transforms.Compose([
        #transforms.RandomCrop(96, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
else:
    transform_train = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.super:
    train_data, test_data = upsample.get_super()
elif args.bicubic:
    train_data, test_data = upsample.get_bicubic()

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
if args.super or args.bicubic:
    trainset.train_data = train_data
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
if args.super or args.bicubic:
    testset.test_data = test_data
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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
    if args.super or args.bicubic:
        net = VGG('VGG16_96') # 22s train, 2.3s test, 10.3% test after 1 epoch
    else:
        net = VGG('VGG16') # 22s train, 2.3s test, 10.3% test after 1 epoch
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

def test(epoch):
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
    test_acc = test(epoch)
    acc_dict["train"].append(train_acc)
    acc_dict["test"].append(test_acc)
    with open(stats_file, 'w', encoding='utf8') as fp:
        json.dump(acc_dict, fp)
