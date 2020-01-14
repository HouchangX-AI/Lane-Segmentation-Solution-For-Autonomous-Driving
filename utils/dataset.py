from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import logging
from PIL import Image
import pandas as pd
import cv2

import torch
from torch.utils.data import Dataset
from utils.process_label import encode_labels


class BasicDataset(Dataset):

    def __init__(self, data_dir, img_size=[1024, 384], crop_offset=690):
        data = pd.read_csv(data_dir)
        self.labels_dir = data['label']
        self.images_dir = data['image']
        self.image_size = img_size

        self.ids = self.images_dir
        np.random.shuffle(self.ids)
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(self, roi_image, roi_label):
        roi_image = roi_image[self.offset:, :]
        roi_label = roi_label[self.offset:, :]
        
        train_image = cv2.resize(roi_image, (self.image_size[0], self.image_size[1]), interpolation=cv2.INTER_LINEAR)
        train_label = cv2.resize(roi_label, (self.image_size[0], self.image_size[1]), interpolation=cv2.INTER_NEAREST)

        train_label = encode_labels(train_label)

        train_image = train_image / (255.0 / 2) - 1

        return train_image, train_label

    def __getitem__(self, i):
        idx = self.ids[i]
        image_path = self.images_dir[idx]
        label_path = self.labels_dir[idx]

        roi_image = cv2.imread(image_path)
        roi_label = cv2.imread(label_path)

        train_img, train_label = self.preprocess(roi_image, roi_label)

        return {'image': torch.from_numpy(train_img), 'mask': torch.from_numpy(train_label)}