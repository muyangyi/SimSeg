import abc
import six
import os
import time
import torch
from simseg.utils.collections import AttrDict

from simseg.core import (
    Hook,
    HookMode,
    get_priority,
    get_hook_mode
)
from simseg.utils import is_list_of, logger, ENV

__all__ = ['BaseRunner']


@six.add_metaclass(abc.ABCMeta)
class BaseRunner:
    """ Basic virtual runner to define interfaces.

    Args:
        cfg (AttrDict): global config.
    """
    def __init__(self, cfg):
        # cfg record
        self.cfg = cfg
        self.state = AttrDict()

        # Build-in protected properties
        self._hooks = []

    @property
    def hooks(self):
        """list[:obj:`Hook`]: A list of registered hooks."""
        return self._hooks

    def register_hook(self, hook, priority='NORMAL', hook_mode='GLOBAL'):
        """Register a hook into the hook list.

        Args:
            hook (:obj:`Hook`): The hook to be registered.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
            hook_mode (int or str or :obj:`HookMode`): Hook mode.
                Controls the work space of the registered hook.
        """

        # Check the hook mode and skip the hook whose mode mis-matches with current simseg mode
        hook_mode = get_hook_mode(hook_mode)
        if hook_mode == HookMode.TRAIN and self.cfg.inference is True:
            return
        if hook_mode == HookMode.VAL and self.cfg.inference is False:
            return

        # Legality check
        assert isinstance(hook, Hook)

        # Get the priority of hook
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority

        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:
                self._hooks.insert(i + 1, hook)
                inserted = True
                break
        if not inserted:
            self._hooks.insert(0, hook)


    def call_hook(self, fn_name, *args):
        for hook in self._hooks:
            if hasattr(hook, fn_name):
                getattr(hook, fn_name)(self, *args)

    @abc.abstractmethod
    def run(self):
        """Start running.
        """
        pass
