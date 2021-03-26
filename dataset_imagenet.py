""" train and test dataset

author baiyu
"""
import os
import sys
import pickle
import random
from skimage import io
import matplotlib.pyplot as plt
import numpy
import torch
from torch.utils.data import Dataset
from PIL import Image


def read_file(file):
    file_list = []
    with open(file, 'r') as fi:
        for line in fi:
            line = line.strip()
            arr = line.split(' ')
            arr[1] = int(arr[1])
            file_list.append(arr)
    return file_list


class ImagenetTrain(Dataset):
    """imagenet 1000 test dataset, derived from
    torch.utils.data.DataSet
    """
    def __init__(self, data_dir, train_file, transform=None):
        self.data_dir = data_dir
        self.file_list = read_file(train_file)
        print(self.data_dir, train_file, len(self.file_list))
        self.transform = transform

    def __len__(self):
        # return len(self.data['fine_labels'.encode()])
        return len(self.file_list)

    def __getitem__(self, index):
        arr = self.file_list[index]
        img_path = os.path.join(self.data_dir, arr[0])
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)
        return arr[1], image

class ImagenetVal(Dataset):
    """imagenet 1000 test dataset, derived from
    torch.utils.data.DataSet
    """
    def __init__(self, data_dir, val_file, transform=None):
        # with open(os.path.join(path, 'test'), 'rb') as cifar100:
        #     self.data = pickle.load(cifar100, encoding='bytes')
        self.data_dir = data_dir
        self.file_list = read_file(val_file)
        print(self.data_dir, val_file, len(self.file_list))
        self.transform = transform

    def __len__(self):
        # return len(self.data['data'.encode()])
        return len(self.file_list)

    def __getitem__(self, index):
        arr = self.file_list[index]
        img_path = os.path.join(self.data_dir, arr[0])
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)
        return arr[1], image

