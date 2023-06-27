import os
from PIL import Image
import numpy as np
import torch
from glob import glob

from simseg.transforms import build_transforms
from simseg.datasets.builder import DATALOADER

__all__ = ["seg"]


class SegDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, dataset_name, data_path, transforms=None) -> None:
        self.cfg = cfg
        self.name = dataset_name
        self.transforms = transforms
        self.data_path = data_path
        
        if dataset_name == "pascal_voc":
            self.root_path = os.path.join(data_path, 'VOCdevkit', 'VOC2012')
            self.image_path = os.path.join(self.root_path, 'JPEGImages')
            self.label_path = os.path.join(self.root_path, 'SegmentationClass')
            list_path = os.path.join(self.root_path, 'ImageSets', 'Segmentation', 'val.txt')
            assert os.path.exists(list_path)
            with open(list_path) as file:
                self.name_list = [line.rstrip() for line in file]
        elif dataset_name == "pascal_context":
            self.root_path = os.path.join(data_path, 'VOCdevkit', 'VOC2010')
            self.image_path = os.path.join(self.root_path, 'JPEGImages')
            self.label_path = os.path.join(self.root_path, 'SegmentationClassContext')
            list_path = os.path.join(self.root_path, 'ImageSets', 'SegmentationContext', 'val.txt')
            assert os.path.exists(list_path)
            with open(list_path) as file:
                self.name_list = [line.rstrip() for line in file]
        elif dataset_name == "coco_stuff":
            self.root_path = os.path.join(data_path, 'coco_stuff164k')
            self.image_path = os.path.join(self.root_path, 'images', 'val2017')
            self.label_path = os.path.join(self.root_path, 'annotations', 'val2017')
            name_list = glob(os.path.join(self.image_path, '*.jpg'))
            self.name_list = [(name.split('/')[-1]).rstrip('.jpg') for name in name_list]
        else:
            raise NotImplementedError("Please verify dataset name.")
        
        self.length = len(self.name_list)

    def __getitem__(self, index):
        item_name = self.name_list[index]

        image_name = os.path.join(self.image_path, item_name) + '.jpg'
        image = Image.open(image_name).convert('RGB')
        image = self.transforms(image)

        if self.name == "coco_stuff":
            item_name = item_name + "_labelTrainIds"
        label_name = os.path.join(self.label_path, item_name) + '.png'
        label = Image.open(label_name)
        label = np.array(label)
        label = torch.tensor(label)

        return image, label

    def __len__(self):
        return self.length


def build_torch_valid_loader(cfg, name, mode='valid', **kwargs):
    transforms = build_transforms(cfg, mode=mode)

    batch_size = cfg.data.batch_size_val

    dataset = SegDataset(cfg=cfg, dataset_name=name, data_path=cfg.data.data_path, transforms=transforms)

    data_loader_train = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return data_loader_train


@DATALOADER.register_obj
def seg(cfg):

    valid_loaders = []
    if cfg.data.enable_valid:
        for name in cfg.data.valid_name:
            valid_loader = build_torch_valid_loader(cfg, name, mode='valid')
            valid_loaders.append(valid_loader)

    return dict(val=valid_loaders)
