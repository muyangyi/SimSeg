import argparse
import os
import torch

from copy import deepcopy
try:
    from apex.parallel import convert_syncbn_model
except ImportError:
    pass

from simseg.core import init_device
from simseg.datasets import DATALOADER
from simseg.models import PIPELINE
from simseg.core import cfg, update_cfg
from simseg.utils import build_from_cfg, ENV

from simseg.tasks.linear_prob.linear_runner import LinearProbRunner
from simseg.tasks.linear_prob.config import task_cfg_init_fn, update_clip_config


def parse_args():
    # Parse args with argparse tool
    parser = argparse.ArgumentParser(description='simseg training')
    parser.add_argument('--cfg', type=str, required=True,
                        help='experiment configure file name')
    parser.add_argument("--local_rank", type=int, default=0)  # Compatibility with torch launch.py
    args, cfg_overrided = parser.parse_known_args()

    # Update config from yaml and argv for override
    update_cfg(task_cfg_init_fn, args.cfg, cfg_overrided, preprocess_fn=update_clip_config)

    # Record the global config and its snapshot (for easy experiment reproduction)
    ENV.cfg = cfg
    ENV.cfg_snapshot = deepcopy(cfg)
    ENV.local_rank = args.local_rank


def main():
    # Configuration: user config updating and global config generating
    parse_args()

    # Initialization: set device, generate global config and inform the user library
    init_device(cfg)

    # Build model
    model = build_from_cfg(cfg.model.name, cfg, PIPELINE).to(ENV.device)

    if cfg.model.syncbn:
        if cfg.dist.name == 'torch':
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        else:
            raise NotImplementedError
    
    # Context building: dataloader
    data_loaders = build_from_cfg(cfg.data.name, cfg, DATALOADER)

    # Runner: building and running
    runner = LinearProbRunner(cfg, data_loaders, model)
    runner.run()


if __name__ == '__main__':
    main()
