import os

import torch
import torchvision.datasets as datasets

from simseg.utils import ENV
from simseg.transforms import build_transforms
from simseg.datasets.builder import DATALOADER

__all__ = ["imagenet_local"]


def build_imagenet_local_train_loader(cfg, names, mode, **kwargs):
    transforms = build_transforms(cfg, mode=mode)

    batch_size = cfg.data.batch_size if mode =='train' else cfg.data.batch_size_val

    batch_size = batch_size // ENV.size

    train_path = os.path.join(cfg.data.data_path, 'train')
    train_dataset = datasets.ImageFolder(train_path, transforms)

    datasampler = torch.utils.data.distributed.DistributedSampler(train_dataset, 
        num_replicas=ENV.size,
        rank=ENV.rank,
        shuffle=True if mode == "train" else False)

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset, sampler=datasampler,
        batch_size=batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=False,
        drop_last=True if mode == "train" else False,
    )
    return data_loader_train


def build_imagenet_local_val_loader(cfg, name, mode='valid', **kwargs):
    transforms = build_transforms(cfg, mode=mode)

    batch_size = cfg.data.batch_size if mode =='train' else cfg.data.batch_size_val
    batch_size = batch_size // ENV.size
    
    val_path = os.path.join(cfg.data.data_path, 'val')
    val_dataset = datasets.ImageFolder(val_path, transforms)

    datasampler = torch.utils.data.distributed.DistributedSampler(val_dataset, 
        num_replicas=ENV.size,
        rank=ENV.rank,
        shuffle=False)

    data_loader_val = torch.utils.data.DataLoader(
        val_dataset, sampler=datasampler,
        batch_size=batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return data_loader_val


@DATALOADER.register_obj
def imagenet_local(cfg):
    train_loader = build_imagenet_local_train_loader(cfg, 'imagenet1k', mode='train')

    valid_loader = build_imagenet_local_val_loader(cfg, 'imagenet1k', mode='valid')

    return dict(train=[train_loader], val=[valid_loader])
