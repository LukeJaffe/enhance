#!/usr/bin/env python3

import torch
import torchvision
import skimage
import numpy as np
import progressbar
from PIL import Image

def super_resolve(arr, model):
    img = Image.fromarray(arr)
    ybr = img.convert('YCbCr')
    y, cb, cr = ybr.split()
    input = torch.autograd.Variable(torchvision.transforms.ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
    input = input.cuda()
    out = model(input)
    out = out.cpu()
    out_img_y = out.data[0].numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
    out_arr = np.array(out_img)#[0:32, 0:32, :]

    return out_arr


def get_super(model_file='model_epoch_100.pth'):
    # Load network
    model = torch.load(model_file)

    # Add Gaussian noise to train data
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    print("Super resolving train data:")
    new_train = np.zeros((50000, 96, 96, 3)).astype(np.uint8)
    with progressbar.ProgressBar(max_value=trainset.train_data.shape[0]) as bar:
        for i,img in enumerate(trainset.train_data):
            new_train[i] = super_resolve(img, model)
            bar.update(i)

    # Add Gaussian noise to test data
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False)
    print("Super resolving test data:")
    new_test = np.zeros((10000, 96, 96, 3)).astype(np.uint8)
    with progressbar.ProgressBar(max_value=testset.test_data.shape[0]) as bar:
        for i,img in enumerate(testset.test_data):
            new_test[i] = super_resolve(img, model)
            bar.update(i)

    return new_train, new_test

def get_bicubic():
    # Add Gaussian noise to train data
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    print("Super resolving train data:")
    new_train = np.zeros((50000, 96, 96, 3)).astype(np.uint8)
    with progressbar.ProgressBar(max_value=trainset.train_data.shape[0]) as bar:
        for i,img in enumerate(trainset.train_data):
            new_train[i] = np.array(Image.fromarray(img).resize((96, 96), Image.BICUBIC))
            bar.update(i)

    # Add Gaussian noise to test data
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False)
    print("Super resolving test data:")
    new_test = np.zeros((10000, 96, 96, 3)).astype(np.uint8)
    with progressbar.ProgressBar(max_value=testset.test_data.shape[0]) as bar:
        for i,img in enumerate(testset.test_data):
            new_test[i] = np.array(Image.fromarray(img).resize((96, 96), Image.BICUBIC))
            bar.update(i)

    return new_train, new_test

def get_labels():
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False)
    return trainset.train_labels, testset.test_labels

if __name__=="__main__":
    train_data, test_data = get_data()
    #train_labels, test_labels = get_labels()
    #from IPython import embed
    #embed()
