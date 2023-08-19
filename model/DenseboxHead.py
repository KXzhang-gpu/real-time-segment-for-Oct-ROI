# -*- coding: UTF-8 -*-
import numpy as np

import torch
from torch import nn

from model.utils.common_block import conv_bn_relu

torch.set_printoptions(precision=8)


def get_xy_ctr_np(score_size, score_offset, total_stride):
    """ generate coordinates on image plane for score map pixels (in numpy)
    """
    batch, fm_height, fm_width = 1, score_size, score_size

    y_list = np.linspace(0., fm_height - 1.,
                         fm_height).reshape(1, fm_height, 1, 1)
    y_list = y_list.repeat(fm_width, axis=2)
    x_list = np.linspace(0., fm_width - 1., fm_width).reshape(1, 1, fm_width, 1)
    x_list = x_list.repeat(fm_height, axis=1)
    xy_list = score_offset + np.concatenate((x_list, y_list), 3) * total_stride
    xy_ctr = np.repeat(xy_list, batch, axis=0).reshape(
        batch, -1,
        2)  # .broadcast([batch, fm_height, fm_width, 2]).reshape(batch, -1, 2)
    xy_ctr = torch.from_numpy(xy_ctr.astype(np.float32))
    return xy_ctr


def get_box(xy_ctr, offsets):
    offsets = offsets.permute(0, 2, 3, 1)  # (B, H, W, C), C=4
    offsets = offsets.reshape(offsets.shape[0], -1, 4)
    xy0 = (xy_ctr[:, :, :] - offsets[:, :, :2])
    xy1 = (xy_ctr[:, :, :] + offsets[:, :, 2:])
    bboxes_pred = torch.cat([xy0, xy1], 2)

    return bboxes_pred


class DenseboxHead(nn.Module):
    _hyper_params = dict(
        total_stride=8,
        score_size=17,
        x_size=303,
        num_conv3x3=3,
        head_conv_bn=[False, False, True],
        conv_weight_std=0.0001,
        input_size_adapt=False,
    )

    def __init__(self):
        super(DenseboxHead, self).__init__()

        self.bi = torch.nn.Parameter(torch.tensor(0.).type(torch.Tensor))
        self.si = torch.nn.Parameter(torch.tensor(1.).type(torch.Tensor))

        self.cls_convs = []
        self.bbox_convs = []
        # make conv layers
        self._make_conv3x3()
        self._make_conv_output()
        self._initialize_conv()
        # params initialize
        self.update_params()

    def forward(self, cls_out, reg_out, x_size=0, raw_output=False):
        # classification head
        total_stride = self._hyper_params["total_stride"]
        num_conv3x3 = self._hyper_params["num_conv3x3"]
        cls = cls_out
        bbox = reg_out

        for i in range(0, num_conv3x3):
            cls = getattr(self, 'cls_conv%d' % (i + 1))(cls)
            bbox = getattr(self, 'bbox_conv%d' % (i + 1))(bbox)

        # classification score
        cls_score = self.cls_score(cls)
        cls_score = cls_score.permute(0, 2, 3, 1)
        cls_score = cls_score.reshape(cls_score.shape[0], -1, 1)
        # center-ness score
        ctr_score = self.ctr_score(cls)
        ctr_score = ctr_score.permute(0, 2, 3, 1)
        ctr_score = ctr_score.reshape(ctr_score.shape[0], -1, 1)
        # regression
        offsets = self.bbox_offsets(bbox)
        offsets = torch.exp(self.si * offsets + self.bi) * total_stride
        fm_ctr = self.fm_ctr.to(offsets.device)
        bbox = get_box(fm_ctr, offsets)

        return cls_score, ctr_score, bbox

    def update_params(self):
        x_size = self._hyper_params["x_size"]
        score_size = self._hyper_params["score_size"]
        total_stride = self._hyper_params["total_stride"]
        score_offset = (x_size - 1 - (score_size - 1) * total_stride) // 2
        ctr = get_xy_ctr_np(score_size, score_offset, total_stride)
        self.fm_ctr = ctr
        self.fm_ctr.require_grad = False

    def _make_conv3x3(self):
        num_conv3x3 = self._hyper_params["num_conv3x3"]
        head_conv_bn = self._hyper_params['head_conv_bn']
        for i in range(0, num_conv3x3):
            cls_conv3x3 = conv_bn_relu(256, 256, 3, 1, 0, has_bn=head_conv_bn[i])
            bbox_conv3x3 = conv_bn_relu(256, 256, 3, 1, 0, has_bn=head_conv_bn[i])
            setattr(self, 'cls_conv%d' % (i + 1), cls_conv3x3)
            setattr(self, 'bbox_conv%d' % (i + 1), bbox_conv3x3)

    def _make_conv_output(self):
        self.cls_score = conv_bn_relu(256, 1, 1, 1, 0, has_bn=False)
        self.ctr_score = conv_bn_relu(256, 1, 1, 1, 0, has_bn=False)
        self.bbox_offsets = conv_bn_relu(256, 4, 1, 1, 0, has_bn=False)

    def _initialize_conv(self):
        num_conv3x3 = self._hyper_params['num_conv3x3']
        conv_weight_std = self._hyper_params['conv_weight_std']

        # initialze head
        conv_list = []
        for i in range(num_conv3x3):
            conv_list.append(getattr(self, 'cls_conv%d' % (i + 1)).conv)
            conv_list.append(getattr(self, 'bbox_conv%d' % (i + 1)).conv)

        conv_list.append(self.cls_score.conv)
        conv_list.append(self.ctr_score.conv)
        conv_list.append(self.bbox_offsets.conv)
        conv_classifier = [self.cls_score.conv]

        pi = 0.01
        bv = -np.log((1 - pi) / pi)
        for ith in range(len(conv_list)):
            # fetch conv from list
            conv = conv_list[ith]
            # torch.nn.init.normal_(conv.weight, std=0.01) # from megdl impl.
            torch.nn.init.normal_(
                conv.weight, std=conv_weight_std)  # conv_weight_std = 0.0001
            # nn.init.kaiming_uniform_(conv.weight, a=np.sqrt(5))  # from PyTorch default implementation
            # nn.init.kaiming_uniform_(conv.weight, a=0)  # from PyTorch default implementation
            if conv in conv_classifier:
                torch.nn.init.constant_(conv.bias, torch.tensor(bv))
            else:
                # torch.nn.init.constant_(conv.bias, 0)  # from PyTorch default implementation
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(conv.weight)
                bound = 1 / np.sqrt(fan_in)
                nn.init.uniform_(conv.bias, -bound, bound)
