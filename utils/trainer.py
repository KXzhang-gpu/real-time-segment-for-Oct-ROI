# -*- coding: UTF-8 -*-
import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import numpy as np

from utils.loss.boxloss import Pairwise_Loss, Projection_Loss
from utils.loss.dice import DiceLoss, dice


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def train_epoch(model, loader, optimizer, scaler, epoch, args):
    model.train()
    if args.stage == 'supervise':
        loss_func1 = torch.nn.BCEWithLogitsLoss()
        loss_func2 = DiceLoss(1)
    elif args.stage == 'weak_supervise':
        loss_func1 = Projection_Loss(1)
        loss_func2 = Pairwise_Loss(1)
    else:
        raise ValueError(r"Can't find the corresponding stage for training")

    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        image, target = batch_data["image"], batch_data["label"]
        image = image.to(torch.device(args.device))
        target = target.to(torch.device(args.device))
        for param in model.parameters():
            param.grad = None

        with autocast(enabled=args.amp):
            predict = model(image)
            if args.stage == 'supervise':
                loss1 = loss_func1(predict, target)
                loss2 = loss_func2(predict, target, softmax=False)
                print('ce_loss: {:.3f} dice_loss: {:.3f}'.format(loss1, loss2))
            elif args.stage == 'weak_supervise':
                loss1 = loss_func1(predict, target)
                loss2 = loss_func2(predict, image, args.threshold)
                print('projection_loss: {:.3f} pairwise_loss: {:.3f}'.format(loss1, loss2))
            loss = 1.0 * loss1 + 1.0 * loss2

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(loss.item()),
                "lr: {:.8f}".format(optimizer.param_groups[0]['lr']),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()
        for param in model.parameters():
            param.grad = None
    return run_loss.avg


def val_epoch(model, loader, epoch, args):
    model.eval()
    start_time = time.time()
    run_acc = AverageMeter()
    if args.stage == 'supervise':
        loss_func1 = torch.nn.BCEWithLogitsLoss()
        loss_func2 = DiceLoss(1)
    elif args.stage == 'weak_supervise':
        loss_func1 = Projection_Loss(1)
        loss_func2 = Pairwise_Loss(1)
    else:
        raise ValueError(r"Can't find the corresponding stage for training")
    run_loss = AverageMeter()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            image, target = batch_data["image"], batch_data["label"]

            image, target = image.cuda(args.rank), target.cuda(args.rank)
            with autocast(enabled=args.amp):
                predict = model(image)
                # loss
                if args.stage == 'supervise':
                    loss1 = loss_func1(predict, target)
                    loss2 = loss_func2(predict, target, softmax=False)
                    print('ce_loss: {:.3f} dice_loss: {:.3f}'.format(loss1, loss2))
                elif args.stage == 'weak_supervise':
                    loss1 = loss_func1(predict, target)
                    loss2 = loss_func2(predict, image, args.threshold)
                    print('projection_loss: {:.3f} pairwise_loss: {:.3f}'.format(loss1, loss2))
                loss = 1.0 * loss1 + 1.0 * loss2

                # sigmoid for dice
                out = torch.sigmoid(predict)

            acc_list = []
            if args.device == 'cuda':
                target = target.cpu().numpy()
                out = out.cpu().detach().numpy()
                loss = loss.cpu()
            else:
                target = target.numpy()
                out = out.numpy()
            for i in range(out.shape[0]):
                out[i] = np.where(out[i] > 0.5, 1, 0)
                acc_list.append(dice(out[i], target[i]))

            avg_acc = np.mean(np.array(acc_list))
            run_acc.update(avg_acc, n=args.batch_size)
            run_loss.update(loss.item(), n=args.batch_size)
            if args.rank == 0:
                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    "dice:", avg_acc,
                    'loss:', loss.numpy(),
                    "time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()
    return run_acc.avg, run_loss.avg


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def save_checkpoint(model, file_path, file_name="model.pth"):
    state_dict = model.state_dict()
    file_path = os.path.join(file_path, file_name)
    torch.save(state_dict, file_path)
    print("Saving checkpoint at:", file_path)


def run_training(model,
                 train_loader,
                 val_loader,
                 optimizer,
                 args,
                 scheduler=None,
                 start_epoch=0):
    """
    used for training model
    Parameters
    ----------
    model
    train_loader
    val_loader
    optimizer
    args
    scheduler
    start_epoch

    Returns
    -------

    """
    early_stopping = EarlyStopping(patience=10, verbose=True)
    spend_time = 0
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()  # using float16 to reduce memory

    val_acc_max = 0.0
    for epoch in range(start_epoch, args.max_epochs):
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(model=model,
                                 loader=train_loader,
                                 optimizer=optimizer,
                                 scaler=scaler,
                                 epoch=epoch,
                                 args=args)  # for training one epoch

        spend_time += time.time() - epoch_time
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),)

            with open(os.path.join(args.logdir, args.model_name+"_log.txt"), mode="a", encoding="utf-8") as f:
                f.write(
                    "Final training:{}/{},".format(epoch, args.max_epochs - 1) + "loss:{}".format(train_loss) + "\n")
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)

        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_avg_acc, run_loss = val_epoch(model=model,
                                              loader=val_loader,
                                              epoch=epoch,
                                              args=args)

            if args.rank == 0:
                print(
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    "acc:", val_avg_acc,
                    'loss:', run_loss,
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                with open(os.path.join(args.logdir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(
                        "Final validation:{}/{},".format(epoch, args.max_epochs - 1) + "dice:{},".format(val_avg_acc)
                        + "loss:{},".format(run_loss) + "\n")
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)

                # save model for each validation epoch
                if args.save_checkpoint:
                    file_path = os.path.join(args.save_path, args.model_name)
                    if not os.path.exists(file_path):
                        os.makedirs(file_path)
                    save_checkpoint(model=model, file_path=file_path, file_name='epoch{}.pth'.format(epoch))

                    if val_avg_acc > val_acc_max:
                        print("best has changed: ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                        val_acc_max = val_avg_acc

                        # save best model
                        save_checkpoint(model=model, file_path=file_path, file_name='best_model.pth')

            # stop training if necessary
            early_stopping(val_avg_acc)
            if early_stopping.early_stop:
                print("Early stop！")
                break

            # learning rate scheduler
            if scheduler is not None:
                scheduler.step(-val_avg_acc)
    print("Training Finished !, Best Accuracy: ", val_acc_max, "Total time: {} s.".format(round(spend_time)))

