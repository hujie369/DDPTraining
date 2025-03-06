#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import time
from copy import deepcopy
import os.path as osp
from tqdm import tqdm

import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch import amp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from utils.data_load import get_dataloaders
from utils.events import LOGGER, write_tblog, NCOLS
from utils.general import de_parallel
from utils.checkpoint import save_checkpoint


class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.max_epoch = args.epochs

        if args.resume:
            self.ckpt = torch.load(args.resume, map_location='cpu')

        self.rank = args.rank
        self.local_rank = args.local_rank
        self.world_size = args.world_size
        self.main_process = self.rank in [-1, 0]
        self.save_dir = args.save_dir

        # get model and optimizer
        model = self.get_model(args, device)
        self.optimizer = self.get_optimizer(args, model)
        self.scheduler = self.get_lr_scheduler(args, self.optimizer)
        # get dataloader
        self.train_loader, self.val_loader = self.get_data_loader(self.args)
        # tensorboard
        self.tblogger = SummaryWriter(
            self.save_dir) if self.main_process else None
        self.start_epoch = 0
        # resume
        if hasattr(self, "ckpt"):
            resume_state_dict = self.ckpt['model'].float().state_dict()
            model.load_state_dict(resume_state_dict, strict=True)  # load
            self.start_epoch = self.ckpt['epoch'] + 1
            self.optimizer.load_state_dict(self.ckpt['optimizer'])
            self.scheduler.load_state_dict(self.ckpt['scheduler'])

        self.model = self.parallel_model(args, model, device)
        self.max_stepnum = len(self.train_loader)
        self.batch_size = args.batch_size

        self.loss_info = ['Epoch', 'lr', 'loss']

    # Training Process

    def train(self):
        try:
            self.before_train_loop()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                self.before_epoch()
                self.train_one_epoch(self.epoch)
                self.after_epoch()
            self.print_infor()

        except Exception as _:
            LOGGER.error('ERROR in training loop or eval/save model.')
            raise
        finally:
            self.train_after_loop()

    # Training loop for each epoch
    def train_one_epoch(self, epoch_num):
        try:
            for self.step, self.batch_data in self.pbar:
                self.train_in_steps(epoch_num, self.step)
                self.print_details()
        except Exception as _:
            LOGGER.error('ERROR in training steps.')
            raise

    # Training one batch data.
    def train_in_steps(self, epoch_num, step_num):
        images = self.batch_data[0].to(self.device, non_blocking=True)
        labels = self.batch_data[1].to(self.device)

        # forward
        with amp.autocast(self.device.type):
            outputs = self.model(images)
            loss = self.compute_loss(outputs, labels)
            # 需要乘上GPU数目, 不然实际梯度会变小
            if self.rank != -1:
                loss *= self.world_size
        # backward
        self.scaler.scale(loss).backward()
        self.loss_items = loss.item() / self.world_size
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

    def after_epoch(self):
        lrs_of_this_epoch = [x['lr'] for x in self.optimizer.param_groups]
        self.scheduler.step()  # update lr

        # 接下来的任务只有主进程进行, 包括对模型进行eval, 保存checkpoint等
        if self.main_process:
            remaining_epochs = self.max_epoch - 1 - self.epoch  # self.epoch is start from 0
            eval_interval = self.args.eval_interval
            is_val_epoch = (remaining_epochs == 0) or (
                (self.epoch + 1) % eval_interval == 0)
            if is_val_epoch:
                self.eval_model()
                self.acc = self.evaluate_results[0]
                self.best_acc = max(self.acc, self.best_acc)
            # save ckpt
            ckpt = {
                'model': deepcopy(de_parallel(self.model)),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'epoch': self.epoch,
                'results': self.evaluate_results,
            }

            save_ckpt_dir = osp.join(self.save_dir, 'weights')
            save_checkpoint(ckpt, (is_val_epoch) and (
                self.acc == self.best_acc), save_ckpt_dir, model_name='last_ckpt')

            del ckpt
            # log for tensorboard
            write_tblog(self.tblogger, self.epoch,
                        self.evaluate_results, lrs_of_this_epoch, self.mean_loss)

    def accuracy(self, output, target, topk=(1,)):
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0,
                                                                keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def eval_model(self):
        # 验证阶段
        self.model.eval()
        # val_loss = 0
        val_top1 = 0
        val_top5 = 0
        num_batches = len(self.val_loader)

        with tqdm(
            self.val_loader, desc=f"Epoch {self.epoch}/{self.max_epoch - 1} [Val]", unit="batch"
        ) as pbar:
            with torch.no_grad():
                for images, labels in pbar:
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device)


<< << << < HEAD
                    with amp.autocast(self.device.type):
== == == =
                    with amp.autocast():
>>>>>> > d141a88(build the initial framework based on YOLOv6)
                        outputs = self.model(images)
                        # loss = self.compute_loss(outputs, labels)

                    # val_loss += loss.item()
                    top1, top5 = self.accuracy(outputs, labels, topk=(1, 5))
                    val_top1 += top1.item()
                    val_top5 += top5.item()

        # val_loss /= num_batches
        val_top1 /= num_batches
        val_top5 /= num_batches

        LOGGER.info(
            f"Epoch: {self.epoch} | top1: {val_top1} | top5: {val_top5}")
        self.evaluate_results = [val_top1, val_top5]

    def before_train_loop(self):
        LOGGER.info('Training start...')
        self.start_time = time.time()
        self.scheduler.last_epoch = self.start_epoch - 1
        self.last_opt_step = -1
        self.scaler = amp.GradScaler(self.device.type)

        self.best_acc, self.acc = 0.0, 0.0
        self.evaluate_results = (0, 0)  # top1, top5
        # resume results
        if hasattr(self, "ckpt"):
            self.evaluate_results = self.ckpt['results']
            self.best_acc = self.evaluate_results[1]

        # 准备损失函数
        self.compute_loss = torch.nn.CrossEntropyLoss()

    def before_epoch(self):
        """ In distributed mode, calling the: meth: `set_epoch` method at
        the beginning of each epoch before creating the: class: `DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used."""
        self.model.train()
        if self.rank != -1:
            # 这里是一个同步点, 其他子进程会等待主进程eval完毕
            self.train_loader.sampler.set_epoch(self.epoch)
        self.mean_loss = torch.zeros(1, device=self.device)
        self.optimizer.zero_grad()

        LOGGER.info(('\n' + '%10s' * 3) % (*self.loss_info,))
        self.pbar = enumerate(self.train_loader)
        if self.main_process:
            self.pbar = tqdm(self.pbar, total=self.max_stepnum, dynamic_ncols=True,
                             bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

    # Print loss after each steps
    def print_details(self):
        if self.main_process:
            self.mean_loss = (self.mean_loss * self.step +
                              self.loss_items) / (self.step + 1)
            self.pbar.set_description(('%10s' + ' %10.4g' + '%10.4g')
                                      % (f'{self.epoch}/{self.max_epoch - 1}',
                                         self.scheduler.get_last_lr()[0], self.mean_loss))

    def print_infor(self):
        if self.main_process:
            LOGGER.info(
                f'\nTraining completed in {(time.time() - self.start_time) / 3600:.3f} hours.')

    # Empty cache if training finished
    def train_after_loop(self):
        if self.device != 'cpu':
            torch.cuda.empty_cache()

    @staticmethod
    def get_data_loader(args):
        if args.dataset == 'imagenet':
            train_loader = get_dataloaders(
                args.data_path, args.rank, args.world_size, args.batch_size // args.world_size, args.workers, mode='train')
            val_loader = None
            if args.rank in [-1, 0]:    # 只有才主进程创建val数据加载器
                val_loader = get_dataloaders(
                    args.data_path, args.rank, args.world_size, args.batch_size // args.world_size, args.workers, mode='val')
            return train_loader, val_loader
        else:
            raise ValueError("only the imagenet is supported currently")

    def get_model(self, args, device):
        """
        先写个resnet50, 后面再看情况进行拓展
        """
        if args.model == 'resnet18':
            model = torchvision.models.resnet18(weights=None).to(device)
        elif args.model == 'resnet50':
            model = torchvision.models.resnet50(weights=None).to(device)
        elif args.model == 'resnet101':
            model = torchvision.models.resnet101(weights=None).to(device)
        else:
            raise ValueError("wrong model name!")
        return model

    @staticmethod
    def parallel_model(args, model, device):
        # If DDP mode
        ddp_mode = device.type != 'cpu' and args.rank != -1
        if ddp_mode:
            model = DDP(model, device_ids=[
                        args.local_rank], output_device=args.local_rank)

        return model

    def get_optimizer(self, args, model):
        # 一些默认参数
        base_batch_size = 256
        weight_decay = 1e-4
        momentum = 0.9

        # batch_size过大时, 梯度估计的方差会减小, 模型在更新参数时更稳定, 可能会更倾向于
        # 拟合数据, 因此权重衰减系数可以按照相同比例进行增加, 抑制过拟合
        accumulate = max(1, round(base_batch_size / args.batch_size))
        weight_decay *= args.batch_size * accumulate / base_batch_size
        # 根据实时的batch size对lr0进行缩放
        lr0 = args.lr0 * (args.batch_size / base_batch_size)

        # 单独对非bn的weight施加正则化
        g_bnw, g_w, g_b = [], [], []
        for v in model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                g_b.append(v.bias)
            if isinstance(v, nn.BatchNorm2d):
                g_bnw.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                g_w.append(v.weight)
        # SGD
        optimizer = torch.optim.SGD(
            g_bnw, lr=lr0, momentum=momentum, nesterov=True)
        optimizer.add_param_group(
            {'params': g_w, 'weight_decay': weight_decay})
        optimizer.add_param_group({'params': g_b})
        del g_bnw, g_w, g_b

        return optimizer

    @staticmethod
    def get_lr_scheduler(args, optimizer):
        milestones = list(map(int, args.milestones.split(',')))
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.1)

        return lr_scheduler
