# -*- coding: UTF-8 -*-
import torch
import argparse

from model.unet_with_resnet import UNet
from utils.dataset.get_loader import get_loader
from utils.trainer import run_training


torch.backends.cudnn.benchmark = True


def make_parser():
    parser = argparse.ArgumentParser(description="Unet segmentation training")
    # trainer settings
    parser.add_argument("--pretrain", default=None, type=str, help="root of pretrain weight")
    parser.add_argument("--max_epochs", default=1, type=int, help="max number of training epochs")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size of each epoch")
    parser.add_argument("--num_workers", default=0, type=int, help="number of workers")
    parser.add_argument("--device", default="cuda", type=str, help="device to be use, cuda or cpu")
    parser.add_argument("--n_classes", default=1, type=int, help="number of output channels")
    parser.add_argument("--stage", default="supervise", type=str, help="the stage of model training")
    parser.add_argument("--val_every", default=1, type=int, help="validation frequency")

    # distributed
    parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
    parser.add_argument("--distributed", action="store_true", help="start distributed training")

    # optimizer
    parser.add_argument("--optim_lr", default=0.5*1e-4, type=float, help="optimization learning rate")
    parser.add_argument("--optimizer", default="adam", type=str, help="optimization algorithm")
    parser.add_argument("--amp", default=True, type=bool, help="use amp for training")
    parser.add_argument("-thr", "--Pairwise_loss_threshold", default="0.9", type=float,
                        help="the threshold using in Pairwise Loss")

    # dataloader settings
    parser.add_argument("--dataset_root", default=r'.\datasets\SA1B', type=str, help="root of training data")

    # save and resume
    parser.add_argument('--model_name', default='Unet_supervise', type=str, help='Name of model to train')
    parser.add_argument("--save_checkpoint", default=True, type=bool, help="save checkpoint during training")
    parser.add_argument("--save_path", default=r".\snapshots", type=str, help="path to save model")
    parser.add_argument("--logdir", default=r".\log", type=str, help="directory to save the tensorboard logs")
    return parser


def main(args):
    # build dataloader
    train_loader, val_loader = get_loader(args)
    print("Batch size is:", args.batch_size, ". Totol epoch:", args.max_epochs)

    # build model
    model = UNet(n_classes=args.n_classes)
    if args.pretrain is not None:
        model.load_state_dict(torch.load(args.pretrain))
    if args.device == 'cuda':
        assert torch.cuda.is_available(), "Please check your devices"
    model.to(args.device)

    # build optimizer and scheduler
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=args.optim_lr, weight_decay=1e-8, momentum=0.99)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)

    torch.autograd.set_detect_anomaly(True)
    run_training(model=model,
                 train_loader=train_loader,
                 val_loader=val_loader,
                 optimizer=optimizer,
                 scheduler=scheduler,
                 args=args)


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    main(args)
