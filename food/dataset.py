from PIL import Image
import os
import os.path
import errno
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data


class Food101(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, train=True,
                 transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.train_data_file = "food-101/numpy/train_data.npy"
        self.train_label_file = "food-101/numpy/train_labels.npy"
        self.test_data_file = "food-101/numpy/test_data.npy"
        self.test_label_file = "food-101/numpy/test_labels.npy"

        # now load the picked numpy arrays
        if self.train:
            self.train_data = np.load(self.train_data_file)
            self.train_labels = np.load(self.train_label_file)
        else:
            self.test_data = np.load(self.test_data_file)
            self.test_labels = np.load(self.test_label_file)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return 20250
        else:
            return 20250

