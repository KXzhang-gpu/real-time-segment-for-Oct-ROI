# -*- coding: UTF-8 -*-
import torch
from torch import nn


class FocalLoss(nn.Module):
    default_hyper_params = dict(
        name="focal_loss",
        background=0,
        ignore_label=-1,
        weight=1.0,
        alpha=0.25,
        gamma=2.0,
    )

    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, predict_data, label_data):
        return predict_data
