import torch
import torch.nn as nn

from simseg.utils import Registry, ENV

LOSS = Registry('loss')


def build_loss(name, param):
    if hasattr(nn, name):
        if name == 'CrossEntropyLoss' and 'weight' in param:
            param['weight'] = torch.tensor(param['weight'], dtype=torch.float32)
        if name == 'CrossEntropyLoss' and 'ignore_index' not in param:
            param['ignore_index'] = ENV.cfg.data.ignore_label
        loss = getattr(nn, name)(**param).to(ENV.device)
    else:
        loss_func = LOSS.get(name)
        if loss_func is None:
            raise KeyError(
                f'{name} is not in the LOSS registry. '
                f'Choose among {list(LOSS.obj_dict.keys())}'
            )
        loss = loss_func(**param).to(ENV.device)
    return loss
