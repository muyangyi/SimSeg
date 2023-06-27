
import math
from bisect import bisect_right
from torch.optim import Optimizer
from .builder import LR

__all__ = ['BaseLRScheduler', 'LambdaLR']


"""
For some reason, pytorch lr scheduler has become a shit mountain. So I decide to write a simple one.

This lr scheduler should be stateless, the lr values should be determined by current global step only. 
so we don't need to save/resotre state to/from checkpoint.

usage:

step_per_epoch = 1000
epoch = 100
optimizer = ...
model = ...

lr_scheduler = cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps = step_per_epoch * 10, # warm up for 10 epoch
    num_training_steps = step_per_epoch * epoch, # train 100 epoch
)

global_step = 0

for i in range(epoch):
    for batch_data in data_loader:
        loss, output = model(batch_data)

        lr_scheduler.set_lrs(global_step) # set lrs before optimizer doing step.

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step += 1
"""

class BaseLRScheduler(object):
    """
    Simplify base lr scheduler

    See:
    https://github.com/pytorch/pytorch/blob/c371542efc31b1abfe6f388042aa3ab0cef935f2/torch/optim/lr_scheduler.py#L22
    """
    
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def get_lrs(self, step):
        pass
    
    def set_lrs(self, step):
        lrs = self.get_lrs(step)
        for i, data in enumerate(zip(self.optimizer.param_groups, lrs)):
            param_group, lr = data
            param_group['lr'] = lr
        
        return lrs
    

class LambdaLR(BaseLRScheduler):

    def __init__(self, optimizer, lr_lambda): 
        super(LambdaLR, self).__init__(optimizer)
        self.lr_lambda = lr_lambda
        self.lr_scale = 0.0
    
    def get_lrs(self, step):
        self.lr_scale = self.lr_lambda(step)
        return [lr*self.lr_scale for lr in self.base_lrs]



"""
Copy from huggingface transformers:
https://github.com/huggingface/transformers/blob/master/src/transformers/optimization.py
"""
        

@LR.register_obj
def constant_schedule(optimizer: Optimizer):
    """
    Create a schedule with a constant learning rate, using the learning rate set in optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    return LambdaLR(optimizer, lambda _: 1)


@LR.register_obj
def constant_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int):
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


@LR.register_obj
def linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda)


@LR.register_obj
def multi_step_schedule_with_warmup(optimizer, num_warmup_steps, milestone_steps, gamma=0.1, **kwargs):

    milestone_steps = sorted(milestone_steps)
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        return gamma ** bisect_right(milestone_steps, current_step)

    return LambdaLR(optimizer, lr_lambda)



@LR.register_obj
def cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda)


@LR.register_obj
def cosine_schedule_with_warmup_min_lr_scale(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, min_lr_scale: float = 0.01):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    assert 0 <= min_lr_scale <= 1.0, "min_lr_scale for cosine_schedule_with_warmup_min_lr_scale should be in [0, 1], but is {}".format(min_lr_scale)

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cur_lr_scale = min_lr_scale + (1.0 - min_lr_scale) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        return max(0.0, cur_lr_scale)

    return LambdaLR(optimizer, lr_lambda)

