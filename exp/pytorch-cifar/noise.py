#!/usr/bin/env python3

import torchvision
import skimage
import numpy as np

def get_data(mean=0, var=0.01, seed=42):
    # Random seed very important!
    np.random.seed(seed)

    # Add Gaussian noise to train data
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False)
    if var > 0:
        for i,img in enumerate(trainset.train_data):
            trainset.train_data[i] = (skimage.util.random_noise(img.astype(float)/255.0, mode='gaussian', mean=mean, var=var)*255.0).astype(np.uint8)

    # Add Gaussian noise to test data
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False)
    if var > 0:
        for i,img in enumerate(testset.test_data):
            testset.test_data[i] = (skimage.util.random_noise(img.astype(float)/255.0, mode='gaussian', mean=mean, var=var)*255.0).astype(np.uint8)

    return trainset.train_data, testset.test_data

def get_labels():
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False)
    return trainset.train_labels, testset.test_labels

if __name__=="__main__":
    train_data, test_data = get_data()
    train_labels, test_labels = get_labels()
    from IPython import embed
    embed()
