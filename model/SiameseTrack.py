# -*- coding: UTF-8 -*-
import torch
from torch import nn

from model.utils.common_block import conv_bn_relu, depthwise_xcorr
from model.AlexNet import AlexNet
from model.DenseboxHead import DenseboxHead


class SiamTrack(nn.Module):
    r"""
    SiamTrack model for tracking
    """

    def __init__(self, backbone=AlexNet(), head=DenseboxHead()):
        super(SiamTrack, self).__init__()
        self.backbone = backbone
        self.head = head
        self._make_convs()

    def forward(self, *args, phase=None):
        if phase is None:
            phase = 'train'

        # used for training this model
        if phase == 'train':
            training_data = args[0]
            return self.train_forward(training_data)

        # used for template feature extraction
        elif phase == 'template':
            template_image = args
            return self.get_feature(template_image)

        # used for tracking in test
        elif phase == 'track':
            search_image, cls_z, reg_z = args
            return self.track_forward(search_image, cls_z, reg_z)
        else:
            raise ValueError("Phase non-implemented.")

    def train_forward(self, training_data):
        template_image = training_data["template_image"]
        search_image = training_data["search_image"]

        # feature extraction
        f_x = self.backbone(search_image)
        f_z = self.backbone(template_image)

        # feature adjustment
        cls_x = self.cls_x(f_x)
        cls_z = self.cls_z(f_z)
        reg_x = self.reg_x(f_x)
        reg_z = self.reg_z(f_z)

        # feature matching
        cls_out = depthwise_xcorr(cls_x, cls_z)
        reg_out = depthwise_xcorr(reg_x, reg_z)

        # head
        cls_score, ctr_score, bbox = self.head(cls_out, reg_out)
        predict_data = dict(
            cls_score=cls_score,
            ctr_score=ctr_score,
            bbox=bbox
        )
        return predict_data

    def get_feature(self, template_image):
        # feature extraction for template image
        f_z = self.backbone(template_image)
        cls_z = self.cls_z(f_z)
        reg_z = self.reg_z(f_z)
        return cls_z, reg_z

    def track_forward(self, search_image, cls_z, reg_z):
        # feature extraction for searching image
        f_x = self.backbone(search_image)
        cls_x = self.cls_x(f_x)
        reg_x = self.reg_x(f_x)

        # feature matching
        cls_out = depthwise_xcorr(cls_x, cls_z)
        reg_out = depthwise_xcorr(reg_x, reg_z)

        # head
        cls_score, ctr_score, bbox = self.head(cls_out, reg_out)
        predict_data = dict(
            cls_score=cls_score,
            ctr_score=ctr_score,
            bbox=bbox
        )
        return predict_data

    def _make_convs(self):
        self.cls_x = conv_bn_relu(256, 256, 3, 1, 0, has_relu=False)
        self.cls_z = conv_bn_relu(256, 256, 3, 1, 0, has_relu=False)
        self.reg_x = conv_bn_relu(256, 256, 3, 1, 0, has_relu=False)
        self.reg_z = conv_bn_relu(256, 256, 3, 1, 0, has_relu=False)


if __name__ == '__main__':
    backbone = AlexNet()
    head = DenseboxHead()
    Tracker = SiamTrack()
    template_image = torch.rand(2, 1, 127, 127)
    search_image = torch.rand(2, 1, 303, 303)
    data = dict(template_image=template_image, search_image=search_image)
    predict_data = Tracker(data, phase='train')
    print(predict_data["bbox"].shape)
