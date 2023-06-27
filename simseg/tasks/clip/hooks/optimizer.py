#!/usr/bin/env python
import re

from simseg.core.hooks import OptimizerHook, log
from simseg.utils import logger

try:
    from apex import amp
except ImportError:
    logger.warning(f'=> ImportError: can not import apex, '
                   f'distribute training with apex will raise error')


class ClipOptimizerHook(OptimizerHook):
    def __init__(self, runner):
        super(ClipOptimizerHook, self).__init__(runner)

    def get_optimizer_grouped_parameters(self, model):
        cfg = self.cfg
        base_lr = cfg.optim.lr.init
        base_wd = cfg.optim.param['weight_decay']

        parameter_groups = []
        for key, value in dict(model.named_parameters()).items():
            if not value.requires_grad:
                continue
            param_group = {'params': [value], 'lr': base_lr, 'weight_decay': base_wd}
            for rule in cfg.optim.param_group_rules.values():
                if not re.search(rule['regex'], key):
                    continue
                param_group.update(rule.get('param', {}))

            logger.emph(key, 'wd:', param_group['weight_decay'], 'lr:', param_group['lr'])
            parameter_groups.append(param_group)

        return parameter_groups

