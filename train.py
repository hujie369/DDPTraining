#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
from logging import Logger
import os
import yaml
import os.path as osp
from pathlib import Path
import sys
import datetime

import torch
import torch.distributed as dist

from utils.general import find_latest_checkpoint, increment_name
from utils.events import LOGGER, save_yaml
from utils.envs import get_envs, select_device, set_random_seed
from core.engine import Trainer


ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(
        description='Pytorch Training', add_help=add_help)
    parser.add_argument('--data-path', default='./data/imagenet',
                        type=str, help='path of dataset')
    parser.add_argument('--dataset', default='imagenet',
                        type=str, help='the name of dataset, including imagenet, cifar10, cifar100')
    parser.add_argument('--model', default='resnet18',
                        type=str, help='the name of model')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='total batch size for all GPUs')
    parser.add_argument('--epochs', default=90, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--lr0', default=0.1, type=float,
                        help='the initial learning rate')
    parser.add_argument('--milestones', default='30,60,80', type=str,
                        help='number of epochs for learning rate decay (default: 30,60,80)')
    parser.add_argument('--workers', default=16, type=int,
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--device', default='0', type=str,
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--eval-interval', default=1,
                        type=int, help='evaluate at every interval epochs')
    parser.add_argument('--output-dir', default='./runs/train',
                        type=str, help='path to save outputs')
    parser.add_argument('--name', default='exp', type=str,
                        help='experiment name, saved to output_dir/name')
    parser.add_argument('--dist_url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--gpu_count', type=int, default=0)
    parser.add_argument('--local_rank', type=int,
                        default=-1, help='DDP parameter')
    parser.add_argument('--resume', nargs='?', const=True,
                        default=False, help='resume the most recent training')
    return parser


def check_and_init(args):
    '''check config files and device.'''
    # check files
    master_process = args.rank == 0 if args.world_size > 1 else args.rank == -1
    if args.resume:
        # args.resume can be a checkpoint file path or a boolean value.
        checkpoint_path = args.resume if isinstance(
            args.resume, str) else find_latest_checkpoint()
        assert os.path.isfile(
            checkpoint_path), f'the checkpoint path is not exist: {checkpoint_path}'
        LOGGER.info(
            f'Resume training from the checkpoint file :{checkpoint_path}')
        resume_opt_file_path = Path(
            checkpoint_path).parent.parent / 'args.yaml'
        if osp.exists(resume_opt_file_path):
            with open(resume_opt_file_path) as f:
                # load args value from args.yaml
                args = argparse.Namespace(**yaml.safe_load(f))
        else:
            LOGGER.warning(f'We can not find the path of {Path(checkpoint_path).parent.parent / "args.yaml"},'
                           f' we will save exp log to {Path(checkpoint_path).parent.parent}')
            LOGGER.warning(
                f'In this case, make sure to provide configuration, such as data, batch size.')
            args.save_dir = str(Path(checkpoint_path).parent.parent)
        # set the args.resume to checkpoint path.
        args.resume = checkpoint_path
    else:
        args.save_dir = str(increment_name(
            osp.join(args.output_dir, args.name)))
        if master_process:
            os.makedirs(args.save_dir)

    # check device
    device = select_device(args.device)
    # set random seed
    set_random_seed(1+args.rank, deterministic=(args.rank == -1))
    # save args
    if master_process:
        save_yaml(vars(args), osp.join(args.save_dir, 'args.yaml'))

    return device, args


def main(args):
    '''main function of training'''
    # Setup
    args.local_rank, args.rank, args.world_size = get_envs()
    device, args = check_and_init(args)
    # reload envs because args was chagned in check_and_init(args)
    args.local_rank, args.rank, args.world_size = get_envs()
    LOGGER.info(f'training args are: {args}\n')
    if args.local_rank != -1:  # if DDP mode
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        LOGGER.info('Initializing process group... ')
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo",
                                init_method=args.dist_url, rank=args.local_rank, world_size=args.world_size, timeout=datetime.timedelta(seconds=7200))

    # Start
    trainer = Trainer(args, device)
    trainer.train()

    # End
    if args.world_size > 1 and args.rank == 0:
        LOGGER.info('Destroying process group... ')
        dist.destroy_process_group()


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
