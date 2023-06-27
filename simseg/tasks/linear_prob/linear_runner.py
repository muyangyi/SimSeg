from simseg.core.hooks import log
import time
import torch
import numpy as np

from addict import Dict as adict
from simseg.utils.collections import AttrDict

from simseg.core import DistHook, HookMode, WandbHook
from simseg.core.hooks.log import LogHook
from simseg.core.runners.builder import RUNNER
from simseg.core.runners.epoch_runner import EpochRunner

from simseg.tasks.clip.hooks import *
from simseg.tasks.linear_prob.hooks import LinearEvalHook

from simseg.utils import logger, ENV


@RUNNER.register_obj
class LinearProbRunner(EpochRunner):
    """ A runner used for linear probing

    Args:
        cfg (adict): global config.
    """

    def __init__(self, cfg, data_loaders, model):
        logger.info("Linear probing runner initiated")
        super(LinearProbRunner, self).__init__(cfg, data_loaders, model)
        self._init_linear_prob_runner()


    def _init_linear_prob_runner(self):
        self.total_steps = self.train_steps * self.max_epochs
        self.warmup_steps = int(self.total_steps * self.cfg.optim.lr.warmup_proportion)
        self.train_type = self.cfg.data.train_type

    def init_hook(self):
        self.register_hook(ClipOptimizerHook(self),
                            priority='very_high', hook_mode=HookMode.TRAIN)

        self.register_hook(DistHook(self),
                           priority='very_high', hook_mode=HookMode.TRAIN)
        self.register_hook(ClipCheckpointHook(self),
                           priority='low', hook_mode=HookMode.TRAIN)
        self.register_hook(LogHook(self),
                           priority='very_low')
                            
        self.register_hook(LinearEvalHook(self),
                            priority='very_low', hook_mode=HookMode.TRAIN)         
        if self.cfg.wandb.enable:
            self.register_hook(WandbHook(self),
                            priority='lowest', hook_mode=HookMode.TRAIN)  

    def input_preprocess(self, batch, mode='train'):
        batch = {k: v.cuda(ENV.device, non_blocking=True) for k,v in batch.items()}
        return batch

    def create_batch_dict(self, batch, mode='train'):
        batch_dict = adict()
        batch_dict['image'], batch_dict['label'] = batch  
        return batch_dict

    def train(self, data_iter, epoch_state, train_steps=None):
        if data_iter is None:
            return
        self.model.train()

        data_iter = data_iter[0]
        self.call_hook('_before_train_epoch', epoch_state)
        for batch in data_iter:
            step_state = adict()
            batch = self.create_batch_dict(batch)
            batch = self.input_preprocess(batch)
            
            if train_steps and epoch_state.inner_step > train_steps:
                break

            self.call_hook('_before_train_step', epoch_state, step_state)
            step_state.batch_output = self.batch_processor(batch)
            self.call_hook('_after_train_step', epoch_state, step_state)

            if self.val_dataloader_list and self.val_interval_steps > 0 and \
                ((self.step + 1) % self.val_interval_steps == 0 or (self.step + 1) == self.total_steps):
                for val_dataloader, val_steps, val_dataset_name in zip(self.val_dataloader_list, self.val_steps_list, self.cfg.data.valid_name):
                    self.val(val_dataloader, val_steps, val_dataset_name)
                self.model.train()

            self.step += 1
            epoch_state.inner_step += 1

        self.call_hook('_after_train_epoch', epoch_state)

    def val(self, data_loader, val_steps=None, val_dataset_name=None):
        if data_loader is None:
            return
        self.model.eval()

        if self.cfg.data.single_eval and ENV.rank != 0: return

        epoch_state = adict()
        epoch_state.inner_step = 0
        epoch_state.data_loader = data_loader
        epoch_state.val_steps = val_steps
        epoch_state.dataset_name = val_dataset_name

        self.call_hook('_before_val_epoch', epoch_state)

        for batch in data_loader:
            # init step state dict
            step_state = adict()
            batch = self.create_batch_dict(batch, mode='valid')
            batch = self.input_preprocess(batch, mode='valid')

            if val_steps and epoch_state.inner_step >= val_steps:
                break
            
            self.call_hook('_before_val_step', epoch_state, step_state)
            with torch.no_grad():
                step_state.batch_output = self.batch_processor(batch, valid=True)
            self.call_hook('_after_val_step', epoch_state, step_state)
            epoch_state.inner_step += 1
        self.call_hook('_after_val_epoch', epoch_state)

    def batch_processor(self, data_batch, valid=False):
        loss = 0
        if valid:
            loss, prediction, label = self.model(data_batch, valid=True)
            output = {'prediction': prediction,
                      'label': label,
                      'loss': loss}
        else:
            if self.scaler:
                with torch.cuda.amp.autocast():
                    loss_dict, train_acc1, train_acc5 = self.model(data_batch)
            else:
                loss_dict, train_acc1, train_acc5 = self.model(data_batch)

            output = {}
            for loss_k, loss_v in loss_dict.items():
                self.state.log_metrics.add_store(loss_k, loss_v)
                output[loss_k] = loss_v
                loss += loss_v
            
            self.state.log_metrics.add_store('train_acc1', train_acc1)
            self.state.log_metrics.add_store('train_acc5', train_acc5)
            self.state.log_metrics.add_store('loss', loss)

            output.update({'loss': loss,
                           'train_acc1': train_acc1,
                           'train_acc5': train_acc5,
                           'lr': self.optimizer.param_groups[0]['lr']})

        # empty cuda cache memory for temperary vairables
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return output

    def run(self):
        """Start running.
        """

        # Logging for start running
        logger.info(f'=> Start Running')

        # data loaders
        train_dataloader = self.train_dataloader
        val_dataloader_list = self.val_dataloader_list
        val_steps_list = self.val_steps_list

        self.call_hook('before_run')

        inner_step = 0
        if self.checkpoint:
            inner_step = self.checkpoint['meta']['inner_step']

        while self.epoch < self.max_epochs:
            # init train epoch state dict
            epoch_state = adict()
            epoch_state.inner_step = inner_step
            epoch_state.data_loader = train_dataloader

            # reset inner_step after first epoch from resume
            inner_step = 0
            self.train(train_dataloader, epoch_state, self.train_steps)
            
            self.epoch += 1

            if self.epoch % self.val_interval == 0 and val_dataloader_list and self.val_interval_steps < 0:
                for val_dataloader, val_steps, val_name in zip(val_dataloader_list, val_steps_list, self.cfg.data.valid_name):
                    try:
                        val_data_iter = val_dataloader.get_iterator(0, 0)
                    except:
                        val_data_iter = val_dataloader   
                    self.val(val_data_iter, val_steps, val_name)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')