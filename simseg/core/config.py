r""" Yaml + AttrDict aided DL configuration design.
"""

import os
import yaml
import copy
from typing import List, Tuple
from simseg.utils.collections import AttrDict
from simseg.utils.misc import is_number_or_bool_or_none

__all__ = ['cfg', 'update_cfg']

cfg = AttrDict()

"""
============================= Basic cfg =============================
基础类 config，配置整体的训练流程

注意：所谓基础配置，就是让人一看名字就明白意义的常用配置，可以直接作为cfg的顶层key，
如:cfg.epoch。
如果不是常用配置，不要在顶层key空间声明（细分task的config可以根据需要声明）
"""
cfg.epoch = None # max epoch to train
cfg.seed = None # random seed
cfg.mae_seed = False # random seed
cfg.inference = False # run in inference mode

# runner
cfg.runner = AttrDict()
cfg.runner.name = None # runner type
cfg.runner.val_interval = 1 # do validation for each X training epoch
cfg.runner.val_interval_steps = -1 # do validation for each X training step. if set, cfg.runner.val_interval will be ignored




"""
====== Core component cfg: model, data, optimizer, distributed ======
模型训练的核心模块，因为不同训练任务这些模块的差异性很大，所以由name来指定相应类型，
并且将不同类型的dataset/model的参数放在不同的子obj上来做key空间分隔。

在这个文件里，不声明具体参数，具体参数在各个task的config中声明。
"""

# Distributed training configurations
cfg.dist = AttrDict()
cfg.dist.name = 'apex'
cfg.dist.enable_adasum = False
cfg.dist.enable_adascale = False
cfg.dist.fp16 = False # 是否用混合精度训练，只在dist.name==torch时生效
cfg.dist.param = dict()

# Model configurations
cfg.model = AttrDict()
cfg.model.name = None

# Dataset configurations
cfg.data = AttrDict()
cfg.data.name = None
cfg.data.batch_size = None
cfg.data.batch_size_val = None
cfg.data.train_steps = None
cfg.data.val_steps = None


# Optimizing configurations
cfg.optim = AttrDict()
cfg.optim.name = 'SGD'
cfg.optim.param = dict(momentum=0.9, weight_decay=1e-4)
cfg.optim.param_group_rules = dict() #TODO: simplify it
cfg.optim.grad_clip = dict()

# optimizer lr
cfg.optim.lr = AttrDict()
cfg.optim.lr.name = 'constant_schedule' # name of lr scheduler
cfg.optim.lr.init = 0.01
cfg.optim.lr.warmup_proportion = 0.1
cfg.optim.lr.warmup_epoch = None
cfg.optim.lr.param = dict()
"""
================== Helper function cfg (基本和hook绑定) ==================
各种辅助类功能的config，如checkpoint，log输出等等。
这些功能一般通过hook实现，注意在这个文件里只声明core模块中实现的通用功能的参数。
"""

# Checkpoint config
cfg.ckpt = AttrDict()
cfg.ckpt.dir = None # The dir for saving checkpoint
cfg.ckpt.step_interval = 500 # step interval for saving checkpoint
cfg.ckpt.filename =  'latest_ckpt.pth'
cfg.ckpt.external_resume = None # external checkpoint file path. If set, resume from it.
cfg.ckpt.auto_resume = True # whether to load from checkpoint automatically
cfg.ckpt.soft_resume = False

# Log config
cfg.log = AttrDict()
cfg.log.interval_train = 100
cfg.log.interval_val = 100

# update cfgs
def update_cfg(task_cfg_init_fn,
               cfg_yaml: str,
               cfg_argv: List[str],
               preprocess_fn=None) -> AttrDict:
    r""" Update cfg with user yaml file and agrv

    Args:
        task_cfg_init_fn: a function to init task specified config. 
        cfg_yaml: the user cfg yaml path, local paths are supported
        cfg_argv: the supplimentary argv from command line
        preprocess_fn: a function to do task specifid cfg preprocessing

    Returns:
        the updated cfg
    """

    # init cfg by task specified config
    task_cfg_init_fn(cfg)

    # Check existence
    if not os.path.exists(cfg_yaml):
        raise ValueError(f'cfg file not found: {cfg_yaml}')

    # Load yaml from disk and update cfg
    with open(cfg_yaml) as f:
        user_cfg = yaml.load(f, Loader=yaml.FullLoader)
        _merge_a_into_b(user_cfg, cfg)

    # update cfg from command line args like "data.batch_size=128"
    _update_cfg_from_argv(cfg_argv)

    # task specified preprocessing
    if preprocess_fn:
        preprocess_fn(cfg)

    # set cfg immutable
    cfg.set_this_dict_immutable(True)

    return cfg


# Update cfg from agrv for override
def _update_cfg_from_argv(cfg_argv: List[str],
                         delimiter: str = '=') -> None:
    r""" Update global cfg with list from argparser

    Args:
        cfg_argv: the config list to be updated, like ['epoch=10', 'save.last=False']
        dilimeter: the dilimeter between key and value of the given config
    """

    def resolve_cfg_with_legality_check(keys: List[str]) -> Tuple[AttrDict, str]:
        r""" Resolve the parent and leaf from given keys and check their legality.

        Args:
            keys: The hierarchical keys of global cfg

        Returns:
            the resolved parent AttrDict obj and its legal key to be upated.
        """

        obj, obj_repr = cfg, 'cfg'
        for idx, sub_key in enumerate(keys):
            if not isinstance(obj, dict) or sub_key not in obj:
                raise ValueError(f'Undefined attribute "{sub_key}" detected for "{obj_repr}"')
            if idx < len(keys) - 1:
                obj = obj.get(sub_key)
                obj_repr += f'.{sub_key}'
        return obj, sub_key

    for str_argv in cfg_argv:
        item = str_argv.split(delimiter, 1)
        assert len(item) == 2, "Error argv (must be key=value): " + str_argv
        key, value = item
        obj, leaf = resolve_cfg_with_legality_check(key.split('.'))
        value = add_quotation_to_string(value)
        value = _decode_cfg_value(value)
        value = _check_and_coerce_cfg_value_type(value, obj[leaf], key)
        obj[leaf] = value


def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, update the
    options in b if they are also specified in a.
    """
    # assert isinstance(a, AttrDict), \
    #     '`a` (cur type {}) must be an instance of {}'.format(type(a), AttrDict)
    assert isinstance(b, AttrDict), \
        '`b` (cur type {}) must be an instance of {}'.format(type(b), AttrDict)

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('Non-existent config key: {}'.format(full_key))

        v = copy.deepcopy(v_)

        # Recursively merge dicts
        if isinstance(b[k], AttrDict):
            assert isinstance(v, dict), f'value for {full_key} must be a dict, got {v} instead'
            stack_push = [k] if stack is None else stack + [k]
            _merge_a_into_b(v, b[k], stack=stack_push)
        else:
            b[k] = v


def add_quotation_to_string(s: str,
                            split_chars: List[str] = None) -> str:
    r""" For eval() to work properly, all string must be added quatation.
         Example: '[[train,3],[val,1]' -> '[["train",3],["val",1]'

    Args:
        s: the original value string
        split_chars: the chars that mark the split of the string

    Returns:
        the quoted value string
    """

    if split_chars is None:
        split_chars = ['[', ']', '{', '}', ',', ' ']
        if '{' in s and '}' in s:
            split_chars.append(':')
    s_mark, marker = s, chr(1)
    for split_char in split_chars:
        s_mark = s_mark.replace(split_char, marker)

    s_quoted = ''
    for value in s_mark.split(marker):
        if len(value) == 0:
            continue
        st = s.find(value)
        if is_number_or_bool_or_none(value):
            s_quoted += s[:st] + value
        elif value.startswith("'") and value.endswith("'") or value.startswith('"') and value.endswith('"'):
            s_quoted += s[:st] + value
        else:
            s_quoted += s[:st] + '"' + value + '"'
        s = s[st + len(value):]

    return s_quoted + s


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, str):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        from ast import literal_eval
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    if value_b is None:
        return value_a

    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, str):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    elif isinstance(value_a, str) and isinstance(value_b, list):
        if value_a.startswith('[') and value_a.endswith(']'):
            value_a = value_a[1:-1]
        value_a = value_a.split(',')
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a
