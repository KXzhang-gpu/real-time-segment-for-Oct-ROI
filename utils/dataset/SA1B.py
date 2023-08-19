# -*- coding: UTF-8 -*-
import os
import json
import pickle

import torch
from torch.utils.data import Dataset
import pycocotools.mask as mask_utils
import numpy as np
import cv2


class SA1B(Dataset):
    """
    SA1B dataset helper
    Parameters:
        dataset_root: str
            path to dataset file
        split: str
            the type of data usage, includes 'train' 'val' and 'test'
        transform: class
            transform used for dataset
    """
    def __init__(self, dataset_root, split, transform=None, update_cache=False):
        super(SA1B, self).__init__()
        self.transform = transform
        self.dataset_root = dataset_root
        self.split = split
        self.image_list = open(os.path.join(self.dataset_root, self.split + '.txt')).readlines()

        # load the data
        self.max_area_factor = 0.8
        self._ensure_cache(update_cache)

    def __len__(self):
        return len(self.data_info_list)

    def __getitem__(self, item):
        record = self.data_info_list[item]
        image_path = record["image_path"]
        label_path = record["label_path"]
        label_id = record["label_id"]

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        with open(label_path) as data_json:
            data = json.load(data_json)
            label_rle = data["annotations"][label_id]["segmentation"]
            label = np.array(mask_utils.decode(label_rle), dtype=np.float32)
            # bbox = data["annotations"][label_id]["bbox"]

        # image and label, shape=(1, H, W)
        # image_range=(0, 255), label_range=(0, 1)
        image = image[np.newaxis, :]
        label = label[np.newaxis, :]
        data = dict(
            image=image,
            label=label
        )
        # data augment
        if self.transform:
            data = self.transform(data)

        return data

    def _ensure_cache(self, update_cache=False):
        self.data_info_list = []
        cache_file = os.path.join(self.dataset_root, "cache/{}.pkl".format(self.split))
        if os.path.exists(cache_file) and not update_cache:
            with open(cache_file, 'rb') as f:
                self.data_info_list = pickle.load(f)
        else:
            for index in self.image_list:
                index = index.strip('\n')
                label_path = os.path.join(self.dataset_root, 'data', index + '.json')
                with open(label_path) as label_json:
                    labels = json.load(label_json)
                    # load image path
                    image_name = labels["image"]["file_name"]
                    image_path = os.path.join(self.dataset_root, 'data', image_name)

                    # index of masks in image
                    for i in range(len(labels["annotations"])):
                        if self._is_filted(labels["image"], labels["annotations"][i]):
                            record = dict(
                                image_path=image_path,
                                label_path=label_path,
                                label_id=i,
                            )
                            self.data_info_list.append(record)
            cache_dir = os.path.dirname(cache_file)
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            with open(cache_file, 'wb') as f:
                pickle.dump(self.data_info_list, f)

    def _is_filted(self, image_info: dict, annotation: dict):
        """
        filt background masks (having a relatively large bbox)
        """
        w, h = image_info["width"], image_info["height"]
        bbox = annotation["bbox"]
        # todo
        bbox_area = bbox[2]*bbox[3]
        total_area = w*h
        if bbox_area > total_area * self.max_area_factor:
            return False
        else:
            return True


import matplotlib.pyplot as plt
from utils.dataset.Transform import Transforms
from time import time

if __name__ == '__main__':
    start_time = time()
    sa1b = SA1B(r'D:\Downloads\OCT\Program\datasets\SA1B', split='test', transform=Transforms(split='train'))
    sa1b2 = SA1B(r'D:\Downloads\OCT\Program\datasets\SA1B', split='test', transform=None)
    idx = 1000
    data = sa1b[idx]
    data2 = sa1b2[idx]
    image = data["image"]
    label = data["label"]
    print(torch.sum(label))
    print(np.sum(data2["label"]))
    plt.subplot(221)
    plt.imshow(image[0], cmap='gray')
    plt.subplot(222)
    plt.imshow(label[0], cmap='gray')
    plt.subplot(223)
    plt.imshow(data2["image"][0], cmap='gray')
    plt.subplot(224)
    plt.imshow(data2["label"][0], cmap='gray')
    plt.show()
    end_time = time()
    print(end_time-start_time)
