import os
import pickle
from glob import glob

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# image transform for train and test


class Transformer():

    __transform_set = [
        transforms.RandomRotation(90),
        transforms.ColorJitter(),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=(5, 9)),
        transforms.RandomInvert()
    ]

    # image transform for train
    __pre_data_transforms = transforms.Compose([
        transforms.RandomApply(__transform_set, p=0.5),
        transforms.Resize((224, 224)),
    ])

    # image transform for valid and test
    __data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def pre_transform(self, img):
        return self.__pre_data_transforms(img)

    def data_transforms(self, img):
        return self.__data_transforms(img)


class MyDataset(Dataset):  # for training
    def __init__(self, path):

        self.data = []
        self.num_classes = 100

        dict = self.__unpickle(path)
        datas, labels = dict[b'data'], dict[b'fine_labels']
        set_length = len(labels)

        for idx in range(set_length):

            img = datas[idx].reshape((1024, 3), order='F').reshape((32, 32, 3))

            label_idx = labels[idx]
            label_vector = np.zeros((self.num_classes))
            label_vector[label_idx] = 1

            self.data.append((img, label_vector, label_idx))

    def __unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx][0]).convert('RGB')
        img = Transformer().data_transforms(img)

        return img, self.data[idx][1], self.data[idx][2]
