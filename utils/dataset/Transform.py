# -*- coding: UTF-8 -*-
from copy import deepcopy

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import numpy as np
import cv2
from PIL import Image


class TransformsBase(object):
    r"""
    Hyper Parameters
    ----------
    crop_factor: float
        crop image size to (h + h * factor, w + w * factor) where w,h is the size of bbox
    radom_crop_scale_ratio: float
        ratio of (w, h) rescale based on bbox before zero crop
    random_crop_shift_ratio: float
        ratio of (x0, y0) shift based on bbox before zero crop
    crop_scale_factor: list
        the range of scale change
    crop_shift_factor: list
        the range of shift change
    add_noise_ratio: float
        ratio of adding noise for image
    add_noise_snr: list
        the range of SNR for pepper noise
    resize: int
        outputsize for image and label
    horizontal_flip_ratio: float
    """

    default_hyper_params = dict(
        # todo how to design transform params
        crop_factor=[0.5, 5],
        radom_crop_shift_ratio=0.6,
        random_crop_scale_ratio=0.6,
        crop_shift_factor=[-0.2, 0.2],
        crop_scale_factor=[0.8, 1.2],
        add_noise_ratio=0.05,
        add_noise_snr=[0.9, 1],
        resize=224,
        horizontal_flip_ratio=0.5,
    )

    def __init__(self):
        super(TransformsBase, self).__init__()
        self.hyper_params = deepcopy(self.default_hyper_params)

    def update(self, **kwargs):
        self.hyper_params.update(kwargs)


class Transforms(TransformsBase):
    def __init__(self, split='train', **kwargs):
        super(Transforms, self).__init__()
        self.split = split
        self.update(**kwargs)

    def __call__(self, data):
        if self.split == 'train':
            data = self.train_trans(data)
        elif self.split == 'test':
            data = self.test_trans(data)
        return data

    def train_trans(self, data):
        transforms = self.build_train_transform()
        data = transforms[0](data)
        image = data["image"]
        label = data["label"]

        seed = torch.random.seed()
        torch.random.manual_seed(seed)
        image = image.transpose(1, 2, 0)
        image = transforms[1](image)
        torch.random.manual_seed(seed)
        label = label.transpose(1, 2, 0)
        label = transforms[2](label)
        data["image"] = image
        data["label"] = label
        return data

    def test_trans(self, data):
        transforms = self.build_test_transform()
        data = transforms[0](data)
        image = data["image"]
        label = data["label"]

        seed = torch.random.seed()
        torch.random.manual_seed(seed)
        image = image.transpose(1, 2, 0)
        image = transforms[1](image)
        torch.random.manual_seed(seed)
        label = label.transpose(1, 2, 0)
        label = transforms[2](label)
        data["image"] = image
        data["label"] = label
        return data

    def build_train_transform(self):
        size = self.hyper_params["resize"]
        horizontal_flip_ratio = self.hyper_params["horizontal_flip_ratio"]

        data_transform = transforms.Compose([
            Init_Crop(**self.hyper_params),
            AddPepperNoise(**self.hyper_params),
            RadomZeroCrop(**self.hyper_params)])

        image_tansform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC),  # 3 is bicubic
            transforms.RandomHorizontalFlip(p=horizontal_flip_ratio),
            transforms.ToTensor(),
            # todo 确定灰度图数据集的正则化参数
            transforms.Normalize(mean=[0.227], std=[0.1935])])

        label_tansform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((size, size), interpolation=InterpolationMode.NEAREST),  # 3 is bicubic
            transforms.RandomHorizontalFlip(p=horizontal_flip_ratio),
            transforms.ToTensor()])

        return [data_transform, image_tansform, label_tansform]

    def build_test_transform(self):
        size = self.hyper_params["resize"]

        data_transform = transforms.Compose([
            Init_Crop(**self.hyper_params),
            RadomZeroCrop(**self.hyper_params)])

        image_tansform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC),  # 3 is bicubic
            transforms.ToTensor(),
            # todo 确定灰度图数据集的正则化参数
            transforms.Normalize(mean=[0.227], std=[0.1935])])

        label_tansform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((size, size), interpolation=InterpolationMode.NEAREST),  # 3 is bicubic
            transforms.ToTensor()])
        return [data_transform, transforms, label_tansform]


class Init_Crop(TransformsBase):
    def __init__(self, **kwargs):
        super(Init_Crop, self).__init__()
        self.update(**kwargs)
        self.factor = self.hyper_params["crop_factor"]

    def __call__(self, data):
        image = data["image"]
        label = data["label"]
        x, y, w, h = cv2.boundingRect((label[0]).astype(np.uint8))
        _, y_max, x_max = image.shape
        factor = np.random.uniform(self.factor[0], self.factor[1])
        x_min = max(0, x - int(w * factor))
        y_min = max(0, y - int(h * factor))
        x = np.random.randint(x_min, x + 1)
        y = np.random.randint(y_min, y + 1)
        w = min(int(w + factor * w), x_max - x)
        h = min(int(h + factor * h), y_max - y)
        image = image[:, y: y + h, x: x + w]
        label = label[:, y: y + h, x: x + w]
        data["image"] = image
        data["label"] = label
        return data


class RadomZeroCrop(TransformsBase):
    def __init__(self, **kwargs):
        super(RadomZeroCrop, self).__init__()
        self.update(**kwargs)
        self.shift_ratio = self.hyper_params["radom_crop_shift_ratio"]
        self.scale_ratio = self.hyper_params["random_crop_scale_ratio"]
        self.shift_factor = self.hyper_params["crop_shift_factor"]
        self.scale_factor = self.hyper_params["crop_scale_factor"]

    def __call__(self, data):
        image = data["image"]
        label = data["label"]
        # x, y, w, h = [int(float_num) for float_num in data["bbox"]]
        x, y, w, h = cv2.boundingRect((label[0]).astype(np.uint8))
        _, y_max, x_max = image.shape
        crop_mask = np.zeros(image.shape)
        _MAX_TRY = 20

        # todo SA1B中过小的mask可能也需要filt
        for i in range(_MAX_TRY + 1):
            if i == _MAX_TRY:
                crop_mask[:, y:y + h, x:x + w] = 1
                break

            x_, y_, w_, h_ = x, y, w, h
            w_min = int(w_ * self.scale_factor[0])
            h_min = int(h_ * self.scale_factor[0])
            if np.random.rand(1) < self.shift_ratio:
                x_rng = np.random.uniform(self.shift_factor[0], self.shift_factor[1])
                y_rng = np.random.uniform(self.shift_factor[0], self.shift_factor[1])
                x_ = max(0, int(x + w * x_rng))
                y_ = max(0, int(y + h * y_rng))
                if x_ + w_min > x_max or y_ + h_min > y_max:
                    pass

            if np.random.rand(1) < self.scale_ratio:
                w_rng = np.random.uniform(self.scale_factor[0], self.scale_factor[1])
                h_rng = np.random.uniform(self.scale_factor[0], self.scale_factor[1])
                w_ = min(int(w * w_rng), x_max - x_)
                h_ = min(int(h * h_rng), y_max - y_)

            crop_mask[:, y_:y_ + h_, x_:x_ + w_] = 1
            label_ = deepcopy(label)
            label_[crop_mask == 0] = 0
            # filt the unvalid mask
            if np.sum(label_) > 500:
                break
            else:
                crop_mask[:] = 0

        image[crop_mask == 0] = 0
        label[crop_mask == 0] = 0
        data["image"] = image
        data["label"] = label
        return data


class AddPepperNoise(TransformsBase):
    """
    add pepper nosie to image
    """

    def __init__(self, **kwargs):
        super(AddPepperNoise, self).__init__()
        self.update(**kwargs)
        self.ratio = self.hyper_params["add_noise_ratio"]
        self.SNR = self.hyper_params["add_noise_snr"]

    def __call__(self, data):
        image = data["image"]
        if np.random.rand(1) < self.ratio:
            image = np.array(image)
            c, h, w = image.shape
            signal_pct = np.random.uniform(self.SNR[0], self.SNR[1])
            noise_pct = (1 - signal_pct)
            mask = np.random.choice((0, 1, 2), size=(1, h, w), p=[signal_pct, noise_pct / 2., noise_pct / 2.])
            mask = np.repeat(mask, c, axis=0)
            image[mask == 1] = 255  # salt
            image[mask == 2] = 0  # pepper
        data["image"] = image
        return data


class ImageAug(TransformsBase):
    def __init__(self):
        super(ImageAug, self).__init__()

    def __call__(self, image):
        return image


if __name__ == '__main__':
    Transforms()
