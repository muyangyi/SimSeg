import os

import pandas as pd
from PIL import Image
import torch
from transformers import AutoTokenizer

from simseg.utils import ENV, logger
from simseg.datasets.clip.utils import process_caption
from simseg.transforms import build_transforms
from simseg.datasets.builder import DATALOADER

__all__ = ["clip"]


class RawImageDataset(torch.utils.data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, cfg, dataset_name, tokenizer, data_path, transforms=None, mode='train') -> None:
        self.cfg = cfg
        self.name = dataset_name
        self.mode = mode
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.target_len = cfg.model.max_length
        self.padding = 'max_length'
        self.data_path = data_path

        if self.mode == 'train':
            self.image_base = os.path.join(data_path, dataset_name, 'train')
            df_path = os.path.join(data_path, dataset_name, 'train_anno.csv')
            assert os.path.exists(df_path)
            self.df = pd.read_csv(df_path)
        else:
            self.image_base = os.path.join(data_path, dataset_name, 'valid')
            df_path = os.path.join(data_path, dataset_name, 'valid_anno.csv')
            assert os.path.exists(df_path)
            self.df = pd.read_csv(df_path)

        self.images = self.df['image']
        self.captions = self.df['caption']
        if self.mode != 'train':
            self.image_ids = self.df['image_id']
            self.caption_ids = self.df['caption_id']

        self.length = len(self.captions)

    def _pad_tok(self, tok_idx):
        return tok_idx[: self.target_len] + [0] * (self.target_len - len(tok_idx))

    def __getitem__(self, index):
        caption = self.captions[index]
        # Convert caption (string) to word ids (with Size Augmentation at training time).
        if self.mode == 'train':
            caption = process_caption(self.tokenizer, caption)

        encoded_caption = self.tokenizer(
            caption, padding=self.padding, truncation=True, max_length=self.target_len
        )
        input_ids, attention_mask = torch.tensor(encoded_caption['input_ids']), torch.tensor(encoded_caption['attention_mask'])

        image_path = self.images[index]
        image_path = os.path.join(self.image_base, image_path)
        image = Image.open(image_path).convert('RGB')
        image = self.transforms(image)
        if self.mode == 'train':
            return image, input_ids, attention_mask, caption
        else:
            image_id, caption_id = self.image_ids[index], self.caption_ids[index]
            return image, input_ids, attention_mask, caption, image_id, caption_id

    def __len__(self):
        return self.length


def build_torch_shuffle_train_loader(cfg, names, mode, **kwargs):
    transforms = build_transforms(cfg, mode=mode)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_encoder.tag)

    batch_size = cfg.data.batch_size if mode =='train' else cfg.data.batch_size_val

    batch_size = batch_size // ENV.size
    
    datasets = []
    for name in names:
        dataset = RawImageDataset(cfg=cfg, dataset_name=name, data_path=cfg.data.data_path, tokenizer=tokenizer, transforms=transforms, mode=mode)
        datasets.append(dataset)

    entire_dataset = torch.utils.data.ConcatDataset(datasets)

    datasampler = torch.utils.data.distributed.DistributedSampler(entire_dataset, 
        num_replicas=ENV.size,
        rank=ENV.rank,
        shuffle=True if mode == "train" else False)

    data_loader_train = torch.utils.data.DataLoader(
        entire_dataset, sampler=datasampler,
        batch_size=batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=False,
        drop_last=True if mode == "train" else False,
    )
    return [data_loader_train]


def build_torch_debias_sequential_train_loader(cfg, names, mode, **kwargs):
    transforms = build_transforms(cfg, mode=mode)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_encoder.tag)

    batch_size = cfg.data.batch_size if mode =='train' else cfg.data.batch_size_val
    batch_size = batch_size // ENV.size
    
    datasets = []
    data_loaders_train = []
    for name in names:
        dataset = RawImageDataset(cfg=cfg, dataset_name=name, data_path=cfg.data.data_path, tokenizer=tokenizer, transforms=transforms, mode=mode)

        datasampler = torch.utils.data.distributed.DistributedSampler(dataset, 
            num_replicas=ENV.size,
            rank=ENV.rank,
            shuffle=True if mode == "train" else False)

        data_loader_train = torch.utils.data.DataLoader(
            dataset, sampler=datasampler,
            batch_size=batch_size,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            drop_last=True if mode == "train" else False,
        )

        datasets.append(dataset)
        data_loaders_train.append(data_loader_train)

    return datasets, data_loaders_train


def build_torch_valid_loader(cfg, name, mode='valid', **kwargs):
    transforms = build_transforms(cfg, mode=mode)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_encoder.tag)

    batch_size = cfg.data.batch_size if mode =='train' else cfg.data.batch_size_val

    batch_size = batch_size // ENV.size
    
    dataset = RawImageDataset(cfg=cfg, dataset_name=name, data_path=cfg.data.data_path, tokenizer=tokenizer, transforms=transforms, mode=mode)

    datasampler = torch.utils.data.distributed.DistributedSampler(dataset, 
        num_replicas=ENV.size,
        rank=ENV.rank,
        shuffle=False)

    data_loader_train = torch.utils.data.DataLoader(
        dataset, sampler=datasampler,
        batch_size=batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return data_loader_train


@DATALOADER.register_obj
def clip(cfg):
    if cfg.data.train_type == 'shuffle':
        train_loader = build_torch_shuffle_train_loader(cfg, cfg.data.train_name, mode='train')
        train_dataset = None
    elif cfg.data.train_type in ['debias', 'sequential']:
        train_dataset, train_loader = build_torch_debias_sequential_train_loader(cfg, cfg.data.train_name, mode='train')
    else:
        raise NotImplementedError

    valid_loaders = []
    if cfg.data.enable_valid:
        for name in cfg.data.valid_name:
            valid_loader = build_torch_valid_loader(cfg, name, mode='valid')
            valid_loaders.append(valid_loader)

    return dict(train=train_loader, train_dataset=train_dataset, val=valid_loaders)
