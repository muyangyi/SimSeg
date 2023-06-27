import os
import random
import numpy as np

import torch
import torch.distributed as distributed
from torch.nn import SyncBatchNorm

from simseg.models import PIPELINE
from simseg.utils import ENV, build_from_cfg
from simseg.utils import (
    is_list_of,
    logger
)

try:
    from apex.parallel import convert_syncbn_model
except ImportError:
    logger.warning(f'=> ImportError: can not import apex, '
                   f'distribute training with apex will raise error')

__all__ = ['init_device', 'init_resume', 'init_model']


def _load_checkpoint(src_path: str, raise_exception: bool = True):
    r"""
    Load checkpoint from local
    """

    if not isinstance(src_path, str):
        return None

    if os.path.exists(src_path):
        return torch.load(src_path, map_location=ENV.device)


def init_device(cfg):
    # Get the Context instance and record the distribution mode
    ENV.dist_mode = cfg.dist.name

    # Random seed setting
    if cfg.seed is not None:
        seed = cfg.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True

    # Distributed scheme initialization
    torch.backends.cudnn.benchmark = True
    if cfg.dist.name in ['apex', 'torch']:
        torch.cuda.set_device(ENV.local_rank)
        distributed.init_process_group(backend='nccl', init_method='env://')

        ENV.rank = distributed.get_rank()
        ENV.size = distributed.get_world_size()
        ENV.device = torch.device(f'cuda:{ENV.local_rank}')

        logger.info(f'=> Device: running distributed training with '
                    f'{cfg.dist.name} DDP, world size:{ENV.size}')
    elif cfg.dist.name is None:
        assert ENV.local_rank == 0, '--np must be 1 when cfg.dist.name is None'
        torch.cuda.set_device(0)
        ENV.device = torch.device(f'cuda:0')
        logger.info('=> Device: running on single process GPU, distributed training disabled')

    # Legality check for batch_size dividable of ENV.size
    if cfg.data.batch_size is not None:
        assert cfg.data.batch_size % ENV.size == 0
    if cfg.data.batch_size_val is not None:
        assert cfg.data.batch_size_val % ENV.size == 0

    # PyTorch version record
    logger.info(f'=> PyTorch Version: {torch.__version__}\n')


def init_resume(cfg):
    checkpoint = None
    if cfg.resume is not None:
        checkpoint = _load_checkpoint(cfg.resume, raise_exception=not cfg.auto_resume)
    if checkpoint is not None:
        logger.info(f'=> Model resume: loaded from {cfg.resume}\n')
    return checkpoint


def init_model(cfg, resume_checkpoint=None):
    # Build model
    logger.info(f'=> Model: {cfg.model.name} with params {cfg.model.param}')
    if cfg.model.backbone.name is not None:
        logger.info(f'   - Backbone: {cfg.model.backbone.name} with params {cfg.model.backbone.param}')
    if cfg.model.head.name is not None:
        logger.info(f'   - Head: {cfg.model.head.name} with params {cfg.model.head.param}')
    if cfg.model.criterion.name is not None:
        logger.info(f'   - Criterion: {cfg.model.criterion.name} with params {cfg.model.criterion.param}. '
                    f'Other settings: prob_type={cfg.model.criterion.prob_type}; '
                    f'cls_loss_type={cfg.model.criterion.cls_loss_type}; '
                    f'reg_loss_type={cfg.model.criterion.reg_loss_type}')
    model = build_from_cfg(cfg.model.name, cfg, PIPELINE)

    # Load pretrained model if resume checkpoint doesn't exist
    if resume_checkpoint is None:
        pretrained_model_loading(cfg, model)


    # Convert BN into SyncBN if necessary
    sync_bn = cfg.model.param.get('sync_bn', False)
    if sync_bn:
        if cfg.dist.name == 'apex':
            model = convert_syncbn_model(model)
        elif cfg.dist.name == 'torch':
            model = SyncBatchNorm.convert_sync_batchnorm(model)

    # Resume model if necessary
    if resume_checkpoint is not None:
        state_dict = resume_checkpoint.get('state_dict', resume_checkpoint)
        model.load_state_dict(state_dict)

    return model.to(ENV.device)


def pretrained_model_loading(cfg, model):
    # load the checkpoint of the pretrained model
    checkpoint = _load_checkpoint(cfg.model.pretrained)
    if checkpoint is None:
        return

    # extract state_dict from the checkpoint
    src_state_dict = checkpoint.get('state_dict', checkpoint)

    # remove the avoid_prefix and avoid_keys from state_dict only when pretrained_strict is False
    pretrained_strict = cfg.model.pretrained_strict
    if pretrained_strict is False:
        avoid_prefix = cfg.model.pretrained_avoid_prefix
        if avoid_prefix is not None:
            if isinstance(avoid_prefix, str):
                avoid_prefix = [avoid_prefix]
            assert is_list_of(avoid_prefix, str)
            for key in list(src_state_dict.keys()):
                if key.startswith(tuple(avoid_prefix)):
                    src_state_dict.pop(key)
            logger.info(f'=> Pretrained: avoid_prefix [{", ".join(avoid_prefix)}] removed from state_dict if exist')
        avoid_keys = cfg.model.pretrained_avoid_keys
        if avoid_keys is not None:
            if isinstance(avoid_keys, str):
                avoid_keys = [avoid_keys]
            assert is_list_of(avoid_keys, str)
            for key in list(src_state_dict.keys()):
                if key in avoid_keys:
                    src_state_dict.pop(key)
            logger.info(f'=> Pretrained: avoid_keys [{", ".join(avoid_keys)}] removed from state_dict if exist')

    # model mapped loading with target_prefix
    target_prefix = cfg.model.pretrained_target_prefix
    if target_prefix is None:
        keys = model.load_state_dict(src_state_dict, strict=pretrained_strict)
    elif target_prefix == 'auto':
        # TODO: the 'auto' target_prefix is an risky patch to deal with the compatibility
        #       between the old `backbone_only` model where backbone and FC heads are
        #       directly saved without prefix. Collate the early version of pretrained
        #       model and remove this mode if early zerovl versions are no longer supported.
        prefix_mapping = dict()
        for key in model.state_dict().keys():
            prefix, name = key.split('.', 1)
            if name in prefix_mapping:
                raise ValueError(f'pretrained loading onto auto prefix failed. Both {prefix} '
                                 f'and {prefix_mapping[name]} prefix has sub-module {name}')
            prefix_mapping[name] = prefix
        for name in list(src_state_dict.keys()):
            if name in prefix_mapping:
                src_state_dict[f'{prefix_mapping[name]}.{name}'] = src_state_dict[name]
                del src_state_dict[name]
        keys = model.load_state_dict(src_state_dict, strict=pretrained_strict)
        logger.info(f'=> Pretrained: the prefix is automatically filled if necessary')
    else:
        sub_model = model
        for p in target_prefix.split('.'):
            assert hasattr(sub_model, p), f'Illegal pretrained_target_prefix {target_prefix}'
            sub_model = getattr(sub_model, p)
        keys = sub_model.load_state_dict(src_state_dict, strict=pretrained_strict)
        logger.info(f'=> Pretrained: the state_dict is loaded to model.{target_prefix}')

    if len(keys.missing_keys) > 0:
        logger.info(f"=> Pretrained: missing_keys [{', '.join(keys.missing_keys)}]")
    if len(keys.unexpected_keys) > 0:
        logger.info(f"=> Pretrained: unexpected_keys [{', '.join(keys.unexpected_keys)}]")
    logger.info(f'=> Pretrained: loaded with strict={pretrained_strict} from {cfg.model.pretrained}\n')
