import abc
import six
import time
import torch
from simseg.utils.collections import AttrDict

from simseg.core.runners.base_runner import BaseRunner
from simseg.utils import logger


__all__ = ['EpochRunner']


@six.add_metaclass(abc.ABCMeta)
class EpochRunner(BaseRunner):
    """ Runner to run training epoch by epoch
    """

    def __init__(self, cfg, data_loaders, model):
        super(EpochRunner, self).__init__(cfg)

        # important pytorch objects
        self.model = model
        self.data_loaders = data_loaders
        self.optimizer = None
        self.checkpoint = None
        self.scaler = None

        # training progress vars
        self.epoch = 0               # current idx of epoch
        self.step = 0                # current overall idx of step
        self.max_epochs = cfg.epoch  # the overall number of epochs
        self.train_steps = None      # the actual training steps per epoch
        self.val_steps = None        # the actual validation steps per epoch
        self.val_interval = cfg.runner.val_interval
        self.val_interval_steps = cfg.runner.val_interval_steps

        # train/val steps limitation
        self.train_steps = self.cfg.data.train_steps
        self.train_dataloader = data_loaders['train']
        if self.train_dataloader is not None:
            if self.train_steps is None or self.train_steps < 1:
                self.train_steps = sum(len(loader) for loader in self.train_dataloader)

        self.val_steps = self.cfg.data.val_steps # int/None/list
        self.val_dataloader_list = data_loaders['val']
        if self.val_dataloader_list is not None:
            if not isinstance(self.val_dataloader_list, list):
                self.val_dataloader_list = [self.val_dataloader_list]

            if self.val_steps:
                if not isinstance(self.val_steps, list):
                    self.val_steps_list = [self.val_steps] * len(self.val_dataloader_list)
                else:
                    assert len(self.val_steps) == len(self.val_dataloader_list), 'length of val_steps_list must equal val_dataloader_list!'
                    self.val_steps_list = self.val_steps
            else:
                self.val_steps_list = [None] * len(self.val_dataloader_list)

            for idx, steps in enumerate(self.val_steps_list):
                if steps is None or steps < 1:
                    self.val_steps_list[idx]  = len(self.val_dataloader_list[idx])
        else:
            self.val_steps_list = []
            self.val_dataloader_list = []


        # Hook initialization and logging
        self.init_hook()
        logger.info('=> Hooks Registration Accomplished')
        for hook in self._hooks:
            logger.info(f'   - {hook.__class__.__name__} registered')

        # init runner hook
        self.call_hook('init_runner')

    def train(self, data_loader, train_steps=None):
        if data_loader is None:
            return
        self.model.train()

        # init epoch state dict
        epoch_state = AttrDict()
        epoch_state.inner_step = 0
        epoch_state.data_loader = data_loader
        self.call_hook('_before_train_epoch', epoch_state)

        for batch in data_loader:
            # init step state dict
            step_state = AttrDict()
            step_state.data_batch = batch

            if train_steps and epoch_state.inner_step > train_steps:
                break

            self.call_hook('_before_train_step', epoch_state, step_state)
            step_state.batch_output = self.batch_processor(batch)
            self.call_hook('_after_train_step', epoch_state, step_state)

            if self.val_dataloader_list and self.val_interval_steps > 0 and (self.step + 1) % self.val_interval_steps == 0:
                for val_dataloader, val_steps in zip(self.val_dataloader_list, self.val_steps_list):
                    self.val(val_dataloader, val_steps)
                self.model.train()

            epoch_state.inner_step += 1
            self.step += 1

        self.call_hook('_after_train_epoch', epoch_state)

    def val(self, data_loader, val_steps=None):
        if data_loader is None:
            return
        self.model.eval()

        # init epoch state dict
        epoch_state = AttrDict()
        epoch_state.inner_step = 0
        epoch_state.data_loader = data_loader
        epoch_state.val_steps = val_steps

        if hasattr(data_loader, 'dataset_name'):
            epoch_state.dataset_name = data_loader.dataset_name
        else:
            epoch_state.dataset_name = None

        self.call_hook('_before_val_epoch', epoch_state)

        for batch in data_loader:
            # init step state dict
            step_state = AttrDict()
            step_state.data_batch = batch

            if val_steps and epoch_state.inner_step >= val_steps:
                break

            self.call_hook('_before_val_step', epoch_state, step_state)
            with torch.no_grad():
                step_state.batch_output = self.batch_processor(batch)
            self.call_hook('_after_val_step', epoch_state, step_state)

            epoch_state.inner_step += 1

        self.call_hook('_after_val_epoch', epoch_state)


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

        while self.epoch < self.max_epochs:

            self.train(train_dataloader, self.train_steps)
            self.epoch += 1

            if self.epoch % self.val_interval == 0 and val_dataloader_list and self.val_interval_steps < 0:
                for val_dataloader, val_steps in zip(val_dataloader_list, val_steps_list):
                    self.val(val_dataloader, val_steps)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    @abc.abstractmethod
    def init_hook(self):
        pass

    @abc.abstractmethod
    def batch_processor(self, data_batch):
        pass

