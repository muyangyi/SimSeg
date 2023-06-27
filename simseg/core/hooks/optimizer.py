from torch.nn.utils import clip_grad_norm_
import torch.optim
import torch.nn as nn
import copy
import re

from .hook import Hook
from simseg.utils import logger

from simseg.core.optimizer import LR, LARS
from simseg.core.hooks.log import LogMetrics


try:
    from apex import amp
except ImportError:
    pass


class OptimizerHook(Hook):
    r"""
    A kind of Hook to execute Optimizer.
    """

    def __init__(self, runner):
        self.cfg = runner.cfg
        self.dist_name = runner.cfg.dist.name
        self.fp16 = runner.cfg.dist.fp16

        # Set the grad_clip
        grad_clip = runner.cfg.optim.grad_clip
        assert isinstance(grad_clip, dict), f'{grad_clip} is of type {type(grad_clip)} while dict is needed'
        self.grad_clip = grad_clip if len(grad_clip) > 0 else None

    def init_runner(self, runner):
        self.optimizer = self.build_optimizer(runner)
        runner.optimizer = self.optimizer
        self.lr_scheduler = self.build_lr_scheduler(runner)

    def clip_grads(self, runner):
        if self.grad_clip is None:
            return
        if self.dist_name == 'apex':
            clip_grad_norm_(amp.master_params(runner.optimizer), **self.grad_clip)
        elif self.dist_name == 'torch' and self.fp16:
            runner.scaler.unscale_(runner.optimizer)
            clip_grad_norm_(runner.model.parameters(), **self.grad_clip)
        else:
            clip_grad_norm_(runner.model.parameters(), **self.grad_clip)

    def before_run(self, runner):
        optimizer_info = ''
        if self.grad_clip is not None:
            optimizer_info += f'grad_clip: {self.grad_clip}; '
        if len(optimizer_info) > 0:
            logger.info(f'=> Optimizer Info: {optimizer_info}')

    def before_train_step(self, runner, epoch_state, step_state):
        # Set learning rate before step
        lrs = self.lr_scheduler.set_lrs(runner.step)

        runner.optimizer.zero_grad()
        # Log lr stuff
        if isinstance(runner.state.log_metrics, LogMetrics):
            runner.state.log_metrics.add_store('lr0', lrs[0])
            if hasattr(self.lr_scheduler, 'lr_scale'):
                runner.state.log_metrics.add_store('lr_scale', self.lr_scheduler.lr_scale)

    def after_train_step(self, runner, epoch_state, step_state):
        # optimizer step
        loss = step_state.batch_output['loss']
        if torch.is_tensor(loss):
            if self.dist_name == 'apex':
                with amp.scale_loss(loss, runner.optimizer) as scaled_loss:
                    scaled_loss.backward()
            elif self.dist_name == 'torch' and self.fp16:
                runner.scaler.scale(loss).backward()
            else:
                loss.backward()

        self.clip_grads(runner)
        
        if self.dist_name =='torch' and self.fp16:
            runner.scaler.step(runner.optimizer)
            runner.scaler.update()
        else:
            runner.optimizer.step()

    ######################  subclass can re-implement these methods  ############
    def build_optimizer(self, runner):
        cfg = self.cfg
        opt_name = cfg.optim.name
        opt_param_groups = self.get_optimizer_grouped_parameters(runner.model)

        logger.info(f'=> Optimizer: {opt_name} Optimizer with state as follows')
        logger.info(f'   {cfg.optim.param}\n')

        # Update optimizer param, note that the initial lr here is actuall a fake one. The real
        # initial lr is given in LRSchedulerHook.build_scheduler() to support multi-scheduler.
        param = copy.deepcopy(cfg.optim.param)
        param.update(lr=cfg.optim.lr.init)
        param.update(params=opt_param_groups)

        # if module has no prefix
        # load from torch.optim
        if opt_name == 'LARS':
            pass
        elif '.' not in opt_name:
            opt_name = 'torch.optim.' + opt_name

            opt_module_name = '.'.join(opt_name.split('.')[:-1])

            # import related modules
            logger.info(f"Importing {opt_module_name}")
            exec(f'import {opt_module_name}')

        optimizer = eval(opt_name)(**param)
        return optimizer

    def build_lr_scheduler(self, runner):
        r""" Generate an lr scheduler.
        """
        cfg = runner.cfg
        name = cfg.optim.lr.name
        num_training_steps = runner.train_steps * runner.max_epochs
        num_warmup_steps = 0
        if cfg.optim.lr.warmup_proportion is not None:
            # warmup by proportion of total training steps
            num_warmup_steps = int(num_training_steps * cfg.optim.lr.warmup_proportion)
        if cfg.optim.lr.warmup_epoch is not None:
            # warmup by epoch
            num_warmup_steps = int(runner.train_steps * cfg.optim.lr.warmup_epoch)

        kwargs = {
            'num_warmup_steps': num_warmup_steps,
            'num_training_steps': num_training_steps,
            **cfg.optim.lr.param
        }

        # for constant lr scheduler
        if 'constant' in name:
            kwargs.pop('num_training_steps')
        if 'warmup' not in name:
            kwargs.pop('num_warmup_steps')
        if 'milestone' in kwargs:
            milestone = kwargs['milestone']
            kwargs['milestone_steps'] = [m * runner.train_steps for m in milestone]

        logger.info(f'=> LRScheduler: {name} scheduler with state as follows')
        logger.info(f'   {kwargs}\n')
        kwargs['optimizer'] = self.optimizer

        lr_scheduler = LR.get(name)(**kwargs)
        return lr_scheduler

    def get_optimizer_grouped_parameters(self, model: nn.Module):
        return model.parameters()
