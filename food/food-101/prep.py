#!/usr/bin/env python2

import os
from PIL import Image
import numpy as np
import json
import progressbar

train_file = 'meta/train.json'
test_file = 'meta/test.json'
class_file = 'meta/classes.txt'
image_dir = 'images'

new_size = (256, 256)

class_to_id = {}
id_to_class = {}
class_id = 0
with open(class_file, 'rb') as fp:
    for line in fp:
        class_name = line.strip()
        class_to_id[class_name] = class_id
        id_to_class[class_id] = class_name
        class_id += 1

decoder_file = 'numpy/decoder.json'
with open(decoder_file, 'wb') as fp:
    json.dump(id_to_class, fp)

def process_test_data():
    test_data_file = 'numpy/test_data.npy'
    test_labels_file = 'numpy/test_labels.npy'

    test_labels = []
    with open(test_file, 'rb') as fp:
        test_dict = json.load(fp)
        # Count images
        num_images = 0
        for i,(class_name,img_files) in enumerate(test_dict.iteritems()):
            num_images += len(img_files)
        test_img_arr = np.zeros((num_images, 256, 256, 3), dtype=np.uint8)
        img_idx = 0
        with progressbar.ProgressBar(max_value=len(class_to_id)) as bar:
            for i,(class_name,img_files) in enumerate(test_dict.iteritems()):
                class_id = class_to_id[class_name]
                for img_file in img_files:
                    test_labels.append(class_id)
                    img_path = os.path.join(image_dir, '{}.jpg'.format(img_file))
                    img = Image.open(img_path).convert("RGB").resize(new_size)
                    arr = np.array(img).astype(np.uint8)
                    test_img_arr[img_idx, :, :, :] = arr
                    img_idx += 1
                bar.update(i)
        test_label_arr = np.array(test_labels)
        np.save(test_labels_file, test_label_arr)
        np.save(test_data_file, test_img_arr)

def process_train_data():
    NUM_SUPER = 500
    train_data_file = 'numpy/train_data.npy'
    train_labels_file = 'numpy/train_labels.npy'
    train_labels = []
    # Load images
    with open(train_file, 'rb') as fp:
        train_dict = json.load(fp)
        # Count images
        num_images = 0
        for i,(class_name,img_files) in enumerate(train_dict.iteritems()):
            num_images += len(img_files) - NUM_SUPER
        train_img_arr = np.zeros((num_images, 256, 256, 3), dtype=np.uint8)
        img_idx = 0
        with progressbar.ProgressBar(max_value=len(class_to_id)) as bar:
            for i,(class_name,img_files) in enumerate(train_dict.iteritems()):
                class_id = class_to_id[class_name]
                for img_file in img_files[NUM_SUPER:]:
                    train_labels.append(class_id)
                    img_path = os.path.join(image_dir, '{}.jpg'.format(img_file))
                    img = Image.open(img_path).convert("RGB").resize(new_size)
                    arr = np.array(img).astype(np.uint8)
                    train_img_arr[img_idx, :, :, :] = arr
                    img_idx += 1
                bar.update(i)
        print "Done loading..."
        train_label_arr = np.array(train_labels)
        print train_label_arr.shape, train_img_arr.shape
        print "Saving..."
        np.save(train_labels_file, train_label_arr)
        np.save(train_data_file, train_img_arr)
        print "Saved."

def process_super_data():
    NUM_SUPER = 500
    super_data_file = 'numpy/super_data.npy'
    super_labels_file = 'numpy/super_labels.npy'
    super_labels = []
    super_images = []
    with open(train_file, 'rb') as fp:
        train_dict = json.load(fp)
        # Count images
        num_images = 0
        for i,(class_name,img_files) in enumerate(train_dict.iteritems()):
            num_images += NUM_SUPER
        train_img_arr = np.zeros((num_images, 256, 256, 3), dtype=np.uint8)
        img_idx = 0
        with progressbar.ProgressBar(max_value=len(class_to_id)) as bar:
            for i,(class_name,img_files) in enumerate(train_dict.iteritems()):
                class_id = class_to_id[class_name]
                for img_file in img_files[:NUM_SUPER]:
                    super_labels.append(class_id)
                    img_path = os.path.join(image_dir, '{}.jpg'.format(img_file))
                    img = Image.open(img_path).convert("RGB").resize(new_size)
                    arr = np.array(img).astype(np.uint8)
                    super_images.append(arr)
                    img_idx += 1
                bar.update(i)
        print "Done loading..."
        super_label_arr = np.array(super_labels)
        super_img_arr = np.array(super_images)
        print super_label_arr.shape, super_img_arr.shape
        print "Saving..."
        np.save(super_labels_file, super_label_arr)
        np.save(super_data_file, super_img_arr)
        print "Saved."

if __name__=='__main__':
    process_test_data()
    #process_train_data()
    process_super_data()
