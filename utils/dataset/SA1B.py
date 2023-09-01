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
        update_cache: bool
            whether update cache for your dataset. if you have changes in data or code, remember to set this parameter
    """
    def __init__(self, dataset_root, split, transform=None, update_cache=False):
        super(SA1B, self).__init__()
        self.transform = transform
        self.dataset_root = dataset_root
        self.split = split
        self.image_list = open(os.path.join(self.dataset_root, self.split + '.txt')).readlines()

        # load the data
        # self.max_area_factor = 0.8
        # self.min_area = 1000
        self._ensure_cache(update_cache)

    def __len__(self):
        return len(self.data_info_list)

    def __getitem__(self, item):
        record = self.data_info_list[item]
        image_path = record["image_path"]
        label_path = record["label_path"]
        label_id = record["label_id"]
        print(image_path)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        with open(label_path) as data_json:
            data = json.load(data_json)
            label_rle = data["annotations"][label_id]["segmentation"]
            label = np.array(mask_utils.decode(label_rle), dtype=np.float32)
            # bbox = data["annotations"][label_id]["bbox"]
            # predicted_iou = data["annotations"][label_id]["predicted_iou"]
            # stability_score = data["annotations"][label_id]["stability_score"]

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

    @staticmethod
    def _is_filted(image_info: dict, annotation: dict) -> bool:
        """
        filt invaild masks (such as background or some trivial objects)
        """
        w, h = image_info["width"], image_info["height"]
        x_b, y_b, w_b, h_b = annotation["bbox"]
        # todo: check the condition is set correctly or not
        bbox_area = w_b * h_b

        # background objects such as sky, river often touch the boundary
        if w_b > 0.95 * w:
            return False
        if h_b > 0.95 * h:
            return False

        # filt masks which are either too small
        if bbox_area < 50000:
            # 224 * 224 = 50176
            return False

        # corner filter
        if x_b <= 10:
            if y_b <= 10 or y_b + h_b >= h - 10:
                return False
        if x_b + w_b >= w - 10:
            if y_b <= 10 or y_b + h_b >= h - 10:
                return False

        return True


import matplotlib.pyplot as plt
from utils.dataset.Transform import Transforms
from time import time

if __name__ == '__main__':
    start_time = time()
    sa1b = SA1B(r'D:\Downloads\OCT\Program\datasets\SA1B', split='test', transform=Transforms(split='train'))
    sa1b2 = SA1B(r'D:\Downloads\OCT\Program\datasets\SA1B', split='test', transform=None)
    idx = 6
    data = sa1b[idx]
    data2 = sa1b2[idx]
    image = data["image"]
    label = data["label"]
    print(len(sa1b))
    print(torch.min(image))
    print(torch.max(image))
    # print(data["predicted_iou"])
    # print(data["stability_score"])
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
