from functools import wraps
from simseg.utils.collections import AttrDict

import torch
import torch.nn as nn

from .misc import Singleton
from .collections import AttrDict

__all__ = ['ENV']


class GlobalContext(Singleton):
    """ Global Context Manager.

    A singleton in charge of mantaining all the environment stuff, including rank, world size, dist.
    """
    # Instances
    _cfg = None

    # Envs
    _dist_mode = None
    _rank = 0
    _size = 1
    _local_rank = 0
    _device = 0
    _loader_type = None
    _iter_checkpoint_info = {}

    @classmethod
    def cls_root_only(cls, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if cls._instance.rank == 0:  # Pay attention that cls._instance.rank != cls.rank
                return func(*args, **kwargs)
        return wrapper

    def root_only(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self._rank == 0:
                return func(*args, **kwargs)

        return wrapper

    def local_root_only(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self._local_rank == 0:
                return func(*args, **kwargs)

        return wrapper

    @property
    def dist_mode(self):
        return self._dist_mode

    @dist_mode.setter
    def dist_mode(self, mode):
        assert mode in ['apex', 'horovod', 'torch', None]
        self._dist_mode = mode

    @property
    def rank(self):
        return self._rank

    @rank.setter
    def rank(self, rank):
        assert isinstance(rank, int) and rank >= 0
        self._rank = rank

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, size):
        assert isinstance(size, int) and size >= 0
        self._size = size

    @property
    def local_rank(self):
        return self._local_rank

    @local_rank.setter
    def local_rank(self, local_rank):
        assert isinstance(local_rank, int) and local_rank >= 0
        self._local_rank = local_rank

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        assert isinstance(device, torch.device)
        self._device = device


    @property
    def loader_type(self):
        return self._loader_type

    @loader_type.setter
    def loader_type(self, x):
        assert x in ['local', 'parquet']
        self._loader_type = x

    @property
    def cfg(self):
        return self._cfg

    @cfg.setter
    def cfg(self, cfg):
        assert isinstance(cfg, AttrDict)
        self._cfg = cfg


ENV = GlobalContext()
