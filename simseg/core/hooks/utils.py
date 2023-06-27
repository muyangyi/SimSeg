import torch
import numpy as np
from enum import Enum, unique
from collections import OrderedDict
from simseg.utils.collections import AttrDict

from simseg.utils import all_reduce

__all__ = ["Priority", "get_priority", "HookMode", "get_hook_mode"]

@unique
class Priority(Enum):
    r"""Hook priority levels.
    +------------+------------+
    | Level      | Value      |
    +============+============+
    | HIGHEST    | 0          |
    +------------+------------+
    | VERY_HIGH  | 10         |
    +------------+------------+
    | HIGH       | 30         |
    +------------+------------+
    | NORMAL     | 50         |
    +------------+------------+
    | LOW        | 70         |
    +------------+------------+
    | VERY_LOW   | 90         |
    +------------+------------+
    | LOWEST     | 100        |
    +------------+------------+
    """

    HIGHEST = 0
    VERY_HIGH = 10
    HIGH = 30
    NORMAL = 50
    LOW = 70
    VERY_LOW = 90
    LOWEST = 100


def get_priority(priority):
    r""" Get hook priority value.

    Args:
        priority (int or str or Priority): Priority.

    Returns:
        int: The priority value.
    """

    if isinstance(priority, int):
        if priority < 0 or priority > 100:
            raise ValueError('priority must be between 0 and 100')
        return priority
    elif isinstance(priority, Priority):
        return priority.value
    elif isinstance(priority, str):
        return Priority[priority.upper()].value
    else:
        raise TypeError('priority must be an integer or Priority enum value')


@unique
class HookMode(Enum):
    r"""
    Hook modes that indicates the mode where a hook works on.
    For example, the lr_scheduler related hook is unnecessary during
    inference, so its mode should be HookMode.VAL
    """

    GLOBAL = 0  # The hook works under all simseg modes
    TRAIN = 1   # The hook only works under training
    VAL = 2     # The hook only works under inference


def get_hook_mode(hook_mode):
    r""" Get hook mode. The mode is limited by the defined HookMode.

    Args:
        hook_mode (int or str or HookMode): Hook mode.

    Returns:
        int: The hook mode value.
    """

    if isinstance(hook_mode, int):
        return HookMode(hook_mode)
    elif isinstance(hook_mode, str):
        return HookMode[hook_mode.upper()]
    elif isinstance(hook_mode, HookMode):
        return hook_mode
    else:
        raise TypeError('hook_mode must be of type int, str or HookMode')