import os
import wandb

from simseg.utils import ENV, logger, AverageMeter
from simseg.utils.collections import AttrDict

from .hook import Hook


class WandbHook(Hook):
    
    def __init__(self, runner):
        self.train_record_keys = runner.cfg.wandb.train_record_keys # for clip is loss, lr, i2t_acc, t2i_acc, should be in batch_outputs
        self._init_hook(runner)

    @ENV.root_only
    def _init_hook(self, runner):
        wandb.login(key=os.environ["WANDB_API_KEY"]) 
        runner.state.wandb_record = AttrDict()

        runner.state.wandb_record.val_record = {} # 需要和evalhook交互
        # print(runner.state.wandb_record,'dddddddd')
        if runner.val_interval_steps > 0:
            self.val_step_type = 'step'
        else:
            self.val_step_type = 'epoch'
        
        self.train_record_meters = {}
        for key in self.train_record_keys:
            self.train_record_meters[key] = AverageMeter()
    
    @ENV.root_only
    def before_run(self, runner):
        if runner.state.get('wandb_id') is None:
            runner.state.wandb_id = wandb.util.generate_id()
        else:
            logger.emph(f"=> Wandb resumed from {runner.state.wandb_id}")

        wandb.init(
            config=runner.cfg,
            id=runner.state.wandb_id,
            project=runner.cfg.wandb.project,
            entity=runner.cfg.wandb.entity,
            name=runner.cfg.data.exp_name,
            resume="allow",
        )
        logger.emph(
            f"=> Wandb name: {runner.cfg.data.exp_name}, id: {runner.state.wandb_id}"
        )

    @ENV.root_only
    def after_train_step(self, runner, epoch_state, step_state):
        for key in self.train_record_keys:
            self.train_record_meters[key].update(step_state.batch_output[key])

        if self.val_step_type == 'step' and self.every_n_steps(runner, runner.val_interval_steps):
            log_dict = {k: v.avg for k, v in self.train_record_meters.items()}
            wandb.log(log_dict, step = runner.step + 1)
            self._reset_train_records()

    @ENV.root_only
    def after_train_epoch(self, runner, epoch_state):
        if self.val_step_type == 'epoch' and self.every_n_epochs(runner, runner.val_interval):
            log_dict = {k: v.avg for k, v in self.train_record_meters.items()}
            wandb.log(log_dict, step = runner.epoch + 1)
            self._reset_train_records()
        return

    @ENV.root_only
    def after_val_epoch(self, runner, epoch_state):
        if self.val_step_type == 'epoch':
            step = runner.epoch
        else:
            step = runner.step + 1
        wandb.log(runner.state.wandb_record.val_record, step=step)
        return

    @ENV.root_only
    def _reset_train_records(self):
        for key in self.train_record_keys:
            self.train_record_meters[key].reset()
