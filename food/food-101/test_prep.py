#!/usr/bin/env python2

import numpy as np

test_data_file = 'numpy/test_data.npy'
test_labels_file = 'numpy/test_labels.npy'
train_data_file = 'numpy/train_data.npy'
train_labels_file = 'numpy/train_labels.npy'
super_data_file = 'numpy/super_data.npy'
super_labels_file = 'numpy/super_labels.npy'

test_data = np.load(test_data_file)
test_labels = np.load(test_labels_file)
print "Test data:"
print test_data.shape
print test_labels.shape

train_data = np.load(train_data_file)
train_labels = np.load(train_labels_file)
print "Train data:"
print train_data.shape
print train_labels.shape

super_data = np.load(super_data_file)
super_labels = np.load(super_labels_file)
print "Super data:"
print super_data.shape
print super_labels.shape
