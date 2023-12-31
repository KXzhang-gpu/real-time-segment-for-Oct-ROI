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
    zero_crop_scale_ratio: float
        ratio of (w, h) rescale based on bbox before zero crop
    zero_crop_shift_ratio: float
        ratio of (x0, y0) shift based on bbox before zero crop
    zero_crop_scale_factor: list
        the range of scale change
    zero_crop_shift: list
        the range of shift change(unit: pixel)
    zero_crop_is_center: bool
        To ensure that the center of new bbox is object we want to segment
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
        zero_crop_shift_ratio=0.6,
        zero_crop_scale_ratio=0.6,
        zero_crop_shift=[-10, 10],
        zero_crop_scale_factor=[1, 1.5],
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
    """
    build tranforms for dataset
    """
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
            Normalization()])

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
            Normalization()])

        label_tansform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((size, size), interpolation=InterpolationMode.NEAREST),  # 3 is bicubic
            transforms.ToTensor()])
        return [data_transform, image_tansform, label_tansform]


class Init_Crop(TransformsBase):
    """
    crop signal object from all instence segment label
    """
    def __init__(self, **kwargs):
        super(Init_Crop, self).__init__()
        self.update(**kwargs)

    def __call__(self, data):
        image = data["image"]
        label = data["label"]
        x, y, w, h = cv2.boundingRect((label[0]).astype(np.uint8))
        _, y_max, x_max = image.shape
        diagonal = int(np.sqrt(h*h + w*w))
        x = max(0, x - diagonal//2)
        y = max(0, y - diagonal//2)
        w = min(int(w + diagonal), x_max - x)
        h = min(int(h + diagonal), y_max - y)
        image = image[:, y: y + h, x: x + w]
        label = label[:, y: y + h, x: x + w]
        data["image"] = image
        data["label"] = label
        return data


class RadomZeroCrop(TransformsBase):
    """
    Set background to 0 according to a random bbox(size, location) to mark the ROI
    """
    def __init__(self, **kwargs):
        super(RadomZeroCrop, self).__init__()
        self.update(**kwargs)
        self.shift_ratio = self.hyper_params["zero_crop_shift_ratio"]
        self.scale_ratio = self.hyper_params["zero_crop_scale_ratio"]
        self.shift_factor = self.hyper_params["zero_crop_shift"]
        self.scale_factor = self.hyper_params["zero_crop_scale_factor"]
        # self.is_center = self.hyper_params["zero_crop_is_center"]

    def __call__(self, data):
        image = data["image"]
        label = data["label"]
        # x, y, w, h = [int(float_num) for float_num in data["bbox"]]
        x, y, w, h = cv2.boundingRect((label[0]).astype(np.uint8))
        # x_center_times2 = 2 * x + w
        # y_center_times2 = 2 * y + h
        _, y_max, x_max = image.shape
        crop_mask = np.zeros(image.shape)
        _MAX_TRY = 20

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
                x_ = max(0, int(x + x_rng))
                y_ = max(0, int(y + y_rng))
                if x_ + w_min > x_max or y_ + h_min > y_max:
                    pass

            if np.random.rand(1) < self.scale_ratio:
                w_rng = np.random.uniform(self.scale_factor[0], self.scale_factor[1])
                h_rng = np.random.uniform(self.scale_factor[0], self.scale_factor[1])
                w_ = min(int(w * w_rng), x_max - x_)
                h_ = min(int(h * h_rng), y_max - y_)

            # if self.is_center:
            #     x_, y_, w_, h_ = self.centralization(x_, y_, w_, h_, x_center_times2, y_center_times2)

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

    @staticmethod
    def centralization(x, y, w, h, x_c_2, y_c_2):
        """
        change the center of a given bbox
        Parameters
        ----------
        x, y, w, h: bbox
        x_c_2, y_c_2: new center times 2
        Returns
        -------
        new centralized bbox
        """
        x_offset = 2 * x + w - x_c_2
        y_offset = 2 * y + h - y_c_2
        if x_offset:
            w = w - x_offset
        if y_offset:
            h = h - y_offset
        return x, y, w, h


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
    """
    not complete yet
    """
    def __init__(self):
        super(ImageAug, self).__init__()

    def __call__(self, image):
        return image


class Normalization(TransformsBase):
    """
    normalize the image to range:(0,1)
    """
    def __init__(self):
        super(Normalization, self).__init__()

    def __call__(self, image):
        i_max = torch.max(image)
        i_min = torch.min(image)
        return (image-i_min) / (i_max - i_min)


if __name__ == '__main__':
    Transforms()
