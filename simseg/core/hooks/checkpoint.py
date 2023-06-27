#!/usr/bin/env python
import os
import re
import time
import yaml
from .hook import Hook

import torch

from simseg.utils import ENV, logger, filter_state
from simseg.version import __version__


def gen_checkpoint(runner, epoch_state, end_of_epoch=False):
    if end_of_epoch:
        inner_step = 0
        epoch = runner.epoch + 1
    else:
        inner_step = epoch_state.inner_step + 1
        epoch = runner.epoch

    meta = dict(time=time.asctime(),
                simseg_version=__version__,
                torch_version=torch.__version__,
                epoch=epoch,
                step=runner.step + 1,
                inner_step=inner_step)

    if runner.state.get('wandb_id', None):
        meta['wandb_id'] = runner.state.wandb_id

    # Get pure model state dict without DDP
    model = runner.model
    if ENV.dist_mode in ['apex', 'torch']:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    state = dict(state_dict=state_dict,
                 optimizer=runner.optimizer.state_dict(),
                 meta=meta)

    if runner.scaler:
        state['scaler'] = runner.scaler.state_dict()
    return state


def get_dist_state_dict(model_state_dict):
    """
    add 'module.' prefix for keys in model
    necessary for apex & torch backend
    """
    logger.emph("Converting checkpoint to dist mode.")
    if ENV.dist_mode in ['apex', 'torch']:
        model_state_dict = {f"module.{k}": v for k, v in model_state_dict.items()}
    return model_state_dict


@ENV.root_only
def create_checkpoint_if_not_exist(save_dir):
    r""" Dump the reproduction cfg and global cfg as yaml files.
    """

    # Establish save.dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)


GLOBAL_CFG_PATH = './global.yaml'


@ENV.root_only
def dump_cfg(cfg, save_dir):
    # Dump the global cfg if necessary
    global_path = GLOBAL_CFG_PATH
    with open(global_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)


class CheckpointHook(Hook):

    def __init__(self, runner):
        cfg = runner.cfg
        self.interval = cfg.ckpt.step_interval
        self.step_ckpt_filename = cfg.ckpt.filename
        self.save_dir = cfg.ckpt.dir
        self.checkpoint_filename = os.path.join(self.save_dir, self.step_ckpt_filename)
        self.save_last = True

    @ENV.root_only
    def after_train_step(self, runner, epoch_state, step_state):
        if self.every_n_steps(runner, self.interval):
            # noted that this saves model, opt, sched and dataset info
            checkpoint = gen_checkpoint(runner, epoch_state)
            torch.save(checkpoint, self.checkpoint_filename)

    @ENV.root_only
    def after_train_epoch(self, runner, epoch_state):
        ckpt = gen_checkpoint(runner, epoch_state, end_of_epoch=True)

        # epoch ckpt
        mxd = max(len(str(runner.max_epochs)), 3)  # max length of epoch
        epoch_filename = os.path.join(self.save_dir,
                                      f'epoch_{str(runner.epoch).zfill(mxd)}.pth')
        torch.save(ckpt, epoch_filename)

        last_filename = os.path.join(self.save_dir, f'latest_ckpt.pth')
        torch.save(ckpt, last_filename)

    def resume_from_external(self, runner):

        # load checkpoint
        cfg = runner.cfg
        if not cfg.ckpt.external_resume:
            return

        checkpoint = torch.load(cfg.ckpt.external_resume, map_location="cpu")
        checkpoint = self.preprocess_checkpoint(checkpoint)
        model_checkpoint = checkpoint['model']

        logger.emph(f'=> Loading pretrained model: {cfg.ckpt.external_resume}\n')

        # load model
        model = runner.model
        checkpoint_renamed, dismatching_keys, missing_keys, unexpected_keys = filter_state(
            model.state_dict(), model_checkpoint, cfg.model.pretrain_prefix_change_list)
        model.load_state_dict(get_dist_state_dict(checkpoint_renamed), strict=not cfg.ckpt.soft_resume)

        if len(dismatching_keys) > 0:
            logger.warning("************* Keys with dismatched shape *************")
            logger.warning(dismatching_keys)
        if len(missing_keys) > 0:
            logger.warning("*************** Keys missing in checkpoint ***************")
            logger.warning(missing_keys)
        if len(unexpected_keys) > 0:
            logger.warning("************** Unexpected keys in checkpoint *************")
            logger.warning(unexpected_keys)

        assert len(dismatching_keys + missing_keys + unexpected_keys) == 0 or cfg.ckpt.soft_resume
        logger.emph(f'=> Loaded pretrained model: {cfg.ckpt.external_resume}\n')

    def before_run(self, runner):
        if runner.cfg.ckpt.auto_resume:
            loaded = self.load_stepckpt(runner)
        else:
            loaded = False
            
        if not loaded:
            self.resume_from_external(runner)

    def load_stepckpt(self, runner):
        create_checkpoint_if_not_exist(self.save_dir)

        # Dump configs into chekpoint dir.
        dump_cfg(runner.cfg, self.save_dir)

        try:
            checkpoint = torch.load(self.checkpoint_filename, map_location="cpu")
        except Exception:
            # Loading checkpoint failed
            return False

        checkpoint = self.preprocess_checkpoint(checkpoint)
        runner.checkpoint = checkpoint

        runner.epoch = checkpoint['meta']['epoch']
        runner.step = checkpoint['meta']['step']

        runner.optimizer.load_state_dict(checkpoint['optimizer'])
        runner.model.load_state_dict(get_dist_state_dict(checkpoint['model']))

        if runner.scaler:
            runner.scaler.load_state_dict(checkpoint['scaler'])

        if checkpoint['meta'].get('wandb_id'):
            runner.state.wandb_id = checkpoint['meta']['wandb_id']

        # Resume logging
        logger.emph(f'=> Step checkpoint resume from {self.checkpoint_filename}')
        logger.emph(f'=> Start training from epoch {runner.epoch} step {runner.step}')

        return True

    ######################  subclass can re-implement these methods  ############

    def preprocess_checkpoint(self, checkpoint: dict):
        return checkpoint
