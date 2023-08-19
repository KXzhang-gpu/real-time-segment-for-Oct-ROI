# -*- coding: UTF-8 -*-

import torch
from torch.utils.data import DataLoader

from utils.dataset.SA1B import SA1B
from utils.dataset.Transform import Transforms


def get_loader(args):
    train_dataset = SA1B(dataset_root=args.dataset_root,
                         split='train',
                         transform=Transforms(split='train'))
    val_dataset = SA1B(dataset_root=args.dataset_root,
                       split='val',
                       transform=Transforms(split='test'))
    train_loader = DataLoader(dataset=train_dataset,
                              sampler=None,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True,
                              drop_last=False)
    val_loader = DataLoader(dataset=val_dataset,
                            sampler=None,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False,
                            drop_last=False)
    return [train_loader, val_loader]
