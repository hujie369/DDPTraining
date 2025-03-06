#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import glob
import math
import torch
import requests
import pkg_resources as pkg
from pathlib import Path


def increment_name(path):
    '''increase save directory's id'''
    path = Path(path)
    sep = ''
    if path.exists():
        path, suffix = (path.with_suffix(
            ''), path.suffix) if path.is_file() else (path, '')
        for n in range(1, 9999):
            p = f'{path}{sep}{n}{suffix}'
            if not os.path.exists(p):
                break
        path = Path(p)
    return path


def find_latest_checkpoint(search_dir='.'):
    '''Find the most recent saved checkpoint in search_dir.'''
    checkpoint_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    return max(checkpoint_list, key=os.path.getctime) if checkpoint_list else ''


def is_parallel(model):
    '''Return True if model's type is DP or DDP, else False.'''
    return type(model) in (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)


def de_parallel(model):
    '''De-parallelize a model. Return single-GPU model if model's type is DP or DDP.'''
    return model.module if is_parallel(model) else model
