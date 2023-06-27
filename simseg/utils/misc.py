r""" Miscellaneous file, some of them copied from mmcv, providing some basic tools.
     - [Docs](https://mmcv.readthedocs.io/en/latest/)
"""

import os
import io
import pwd
import six
import time
import random
import math
import torch
import socket
import requests
import functools
import itertools
import subprocess
import json
from itertools import repeat
from functools import lru_cache, partial
from importlib import import_module
from collections import abc as container_abcs
from collections.abc import Iterable, Sequence

__all__ = [
    'chunk',
    'multi_apply',
    'is_list_of',
    'is_seq_of',
    'is_tuple_of',
    'is_str',
    'is_number_or_bool_or_none',
    'AverageMeter',
    'Singleton',
    'username',
    'ip',
    'host_info',
    'file_handler',
    'get_time_str',
    'get_free_disk_space',
    'split_list_evenly',
    'flatten_list',
    '_single',
    '_pair',
    '_triple',
    '_quadruple',
    'clever_format',
    'download',
    'download_to_local_file',
    'save_step_info',
    'update_step_ckpt_info',
    'calc_topk_accuracy',
]


def chunk(iterable, chunk_size):
    ret = []
    for record in iterable:
        ret.append(record)
        if len(ret) == chunk_size:
            yield ret
            ret = []
    if ret:
        yield ret


def multi_apply(func, *args, **kwargs):
    r"""Apply function to a list of arguments.
    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.
    Args:
        func (Function): A function that will be applied to a list of
            arguments
    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """

    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def is_str(x):
    r"""
    Whether the input is an string instance.
    """

    return isinstance(x, six.string_types)


def is_number_or_bool_or_none(x: str):
    r""" Return True if the given str represents a number (int or float) or bool
    """

    try:
        float(x)
        return True
    except ValueError:
        return x in ['True', 'False', 'None']


def step_cast(inputs, dst_type, return_type=None):
    r"""Cast elements of an iterable object into some type.

    Args:
        inputs (Iterable): The input object.
        dst_type (type): Destination type.
        return_type (type, optional): If specified, the output object will be
            converted to this type, otherwise an iterator.

    Returns:
        iterator or specified type: The converted object.
    """
    if not isinstance(inputs, Iterable):
        raise TypeError('inputs must be an iterable object')
    if not isinstance(dst_type, type):
        raise TypeError('"dst_type" must be a valid type')

    out_iterable = six.moves.map(dst_type, inputs)

    if return_type is None:
        return out_iterable
    else:
        return return_type(out_iterable)


def list_cast(inputs, dst_type):
    r"""Cast elements of an iterable object into a list of some type.

    A partial method of :func:`step_cast`.
    """
    return step_cast(inputs, dst_type, return_type=list)


def tuple_cast(inputs, dst_type):
    r"""Cast elements of an iterable object into a tuple of some type.

    A partial method of :func:`step_cast`.
    """
    return step_cast(inputs, dst_type, return_type=tuple)


def is_seq_of(seq, expected_type, seq_type=None):
    r"""Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.

    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_list_of(seq, expected_type):
    r"""Check whether it is a list of some type.

    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=list)


def is_tuple_of(seq, expected_type):
    r"""Check whether it is a tuple of some type.

    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=tuple)


def slice_list(in_list, lens):
    r"""Slice a list into several sub lists by a list of given length.

    Args:
        in_list (list): The list to be sliced.
        lens(int or list): The expected length of each out list.

    Returns:
        list: A list of sliced list.
    """
    if not isinstance(lens, list):
        raise TypeError('"indices" must be a list of integers')
    elif sum(lens) != len(in_list):
        raise ValueError(
            'sum of lens and list length does not match: {} != {}'.format(
                sum(lens), len(in_list)))
    out_list = []
    idx = 0
    for i in range(len(lens)):
        out_list.append(in_list[idx:idx + lens[i]])
        idx += lens[i]
    return out_list


def concat_list(in_list):
    r"""Concatenate a list of list into a single list.

    Args:
        in_list (list): The list of list to be merged.

    Returns:
        list: The concatenated flat list.
    """
    return list(itertools.chain(*in_list))


def check_prerequisites(
        prerequisites,
        checker,
        msg_tmpl='Prerequisites "{}" are required in method "{}" but not '
        'found, please install them first.'):
    r"""A decorator factory to check if prerequisites are satisfied.

    Args:
        prerequisites (str of list[str]): Prerequisites to be checked.
        checker (callable): The checker method that returns True if a
            prerequisite is meet, False otherwise.
        msg_tmpl (str): The message template with two variables.

    Returns:
        decorator: A specific decorator.
    """

    def wrap(func):

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            requirements = [prerequisites] if isinstance(
                prerequisites, str) else prerequisites
            missing = []
            for item in requirements:
                if not checker(item):
                    missing.append(item)
            if missing:
                print(msg_tmpl.format(', '.join(missing), func.__name__))
                raise RuntimeError('Prerequisites not meet.')
            else:
                return func(*args, **kwargs)

        return wrapped_func

    return wrap


def _check_py_package(package):
    try:
        import_module(package)
    except ImportError:
        return False
    else:
        return True


def _check_executable(cmd):
    if subprocess.call('which {}'.format(cmd), shell=True) != 0:
        return False
    else:
        return True


def requires_package(prerequisites):
    return check_prerequisites(prerequisites, checker=_check_py_package)


def requires_executable(prerequisites):
    return check_prerequisites(prerequisites, checker=_check_executable)


def username():
    return pwd.getpwuid(os.getuid())[0]


def ip():
    return socket.gethostbyname(socket.gethostname())


def host_info():
    return '{}@{}'.format(username(), ip())


@lru_cache(1000)
def file_handler(fn, mode):
    return open(fn, mode)


def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def get_free_disk_space(path='/'):
    r""" Get free disk space of given path, return it in form of GB """
    disk = os.statvfs(path)
    return disk.f_bavail * disk.f_frsize / 1024 ** 3


def split_list_evenly(x: list,
                      group: int,
                      shuffle: bool = False):
    r"""
    Split the given list into several groups equally. For element number of
    the sub-list sub_len, max(sub_len) <= min(sub_len) + 1.
    """

    assert isinstance(x, list) and isinstance(group, int)
    assert 1 <= group <= len(x), f'Can not split list of {len(x)} into {group} parts'
    if shuffle is True:
        random.shuffle(x)
    return [x[i::group].copy() for i in range(group)]


def flatten_list(x: list):
    r"""
    Flatten a given list recursively
    """

    x_flatten = []
    for term in x:
        if isinstance(term, list):
            x_flatten.extend(flatten_list(term))
        else:
            x_flatten.append(term)
    return x_flatten


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


def clever_format(nums, format="%.2f"):
    if not isinstance(nums, Iterable):
        nums = [nums]
    clever_nums = []

    for num in nums:
        if num > 1e12:
            clever_nums.append(format % (num / 1e12) + "T")
        elif num > 1e9:
            clever_nums.append(format % (num / 1e9) + "G")
        elif num > 1e6:
            clever_nums.append(format % (num / 1e6) + "M")
        elif num > 1e3:
            clever_nums.append(format % (num / 1e3) + "K")
        else:
            clever_nums.append(format % num + "B")

    clever_nums = clever_nums[0] if len(clever_nums) == 1 else (*clever_nums, )

    return clever_nums


class AverageMeter(object):
    r"""
    Computes and stores the average and current value
    Imported from https://github.com/pytorch/experiment/blob/master/imagenet/train.py#L247-L262
    """

    def __init__(self):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Singleton(object):
    r"""
    Singleton base class
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls._instance


def download(url, timeout=20, retry=3):
    finish_download = False
    time_to_try = retry
    content = None
    while time_to_try and not finish_download:
        rsp = requests.get(url, timeout=timeout)
        if rsp.status_code == requests.codes.ok:
            finish_download = True
            content = rsp.content
        else:
            time_to_try -= 1
    if finish_download and content is not None:
        return content
    else:
        raise Exception('download {} failed'.format(url))


def download_to_local_file(url, local_path, timeout=20, retry=3):
    content = download(url, timeout=timeout, retry=retry)
    with open(local_path, 'wb') as f:
        f.write(content)


def save_step_info(save_dict, save_path):
    with io.open(save_path, 'w') as f:
        f.write("{}\n".format(json.dumps(save_dict)))


def load_dict_from_file(file):
    with io.open(file) as f:
        file_info = f.read()
    if isinstance(file_info, bytes):
        file_info = file_info.decode("utf-8")
    file_info = file_info.strip().split("\n")[0]
    file_info = json.loads(file_info)
    return file_info


def update_step_ckpt_info(cfg):
    info = {'previous_step': 0, 'epoch': 0}
    step_info_path = os.path.join(cfg.save.dir, 'step_info')
    if not os.path.exists(step_info_path):
        return info
    step_info = load_dict_from_file(step_info_path)
    return step_info

def calc_topk_accuracy(output, target, topk=(1,)):
    '''
    Modified from: https://gist.github.com/agermanidis/275b23ad7a10ee89adccf021536bb97e
    Given predicted and ground truth labels,
    calculate top-k accuracies.
    '''
    maxk = max(topk)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.div_(torch.sum(target >= 0)))
    return res