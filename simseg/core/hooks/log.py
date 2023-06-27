from collections import OrderedDict
import time
import datetime

from .hook import Hook
from simseg.utils import logger

def timedelta_format(td):
    d = td.days
    sec = td.seconds
    h = sec // 3600
    m = sec % 3600 // 60
    s = sec % 60
    format_str = ''
    if d:
        format_str += f'{d}d'
    if h:
        format_str += f'{h}h'
    if m:
        format_str += f'{m}m'
    format_str += f'{s}s'
    return format_str

class LogMetrics(object):
    def __init__(self):
        self.counter_dict = OrderedDict()
        self.store_dict = OrderedDict()
        self.start_time_ns = time.time_ns()
    
    def add_counter(self, key, value):
        """
        类似于metrics中的counter，常用来监控吞吐
        """
        values = self.counter_dict.get(key, [])
        values.append(value)
        self.counter_dict[key] = values
    
    def add_store(self, key, value):
        """
        类似于metrics中的store，常用来监控一些数值
        """
        values = self.store_dict.get(key, [])
        values.append(value) 
        self.store_dict[key] = values

    def reset(self):
        self.counter_dict = OrderedDict()
        self.store_dict = OrderedDict()
        self.start_time_ns = time.time_ns()
    
    def get_counter_info(self):
        time_elapsed = float(time.time_ns() - self.start_time_ns) / 1e9
        counter_info = OrderedDict()
        for key, values in self.counter_dict.items():
            counter_info[key] = float(sum(values)) / time_elapsed
        return counter_info
    
    def get_store_info(self):
        store_info = OrderedDict()
        for key, values in self.store_dict.items():
            store_info[key] = float(sum(values)) / len(values)
        return store_info


class LogHook(Hook):

    def __init__(self, runner):
        runner.state.log_metrics = LogMetrics()
        self.last_step = 0
        self.last_step_timestamp = 0


    def before_epoch(self, runner, epoch_state):
        # 在 epoch 开始时重置 metrics 记录
        runner.state.log_metrics.reset()
        self.last_step = 0
        now = time.time_ns()
        self.last_step_timestamp = now
        self.epoch_start_timestamp = now


    def get_metrics_str(self, log_metrics, epoch_state):
        items = []
        for key, value in log_metrics.get_store_info().items():
            if value < 1e-3:
                items.append(f'{key}: {value:.3e}')
            else:
                items.append(f'{key}: {value:.4f}')
        for key, value in log_metrics.get_counter_info().items():
            items.append(f'{key}: {value:.3f}/s')
        # step time
        time_elapsed = (time.time_ns() - self.last_step_timestamp) / 1e9
        step_time = time_elapsed / max((epoch_state.inner_step - self.last_step), 1)
        items.append(f'step_time: {step_time:.3f} s')
        return ' | '.join(items)
    
    def after_train_step(self, runner, epoch_state, step_state):
        log_interval = runner.cfg.log.interval_train
        if self.every_n_inner_steps(epoch_state, log_interval):
            progress = f'Epoch [{runner.epoch+1}/{runner.max_epochs}]' \
                    f'[{epoch_state.inner_step+1}/{runner.train_steps}]'
            metrics_str = self.get_metrics_str(runner.state.log_metrics, epoch_state)
            log_str = f'{progress} {metrics_str}'
    
            logger.info(log_str)

            runner.state.log_metrics.reset()
            self.last_step = epoch_state.inner_step
            self.last_step_timestamp = time.time_ns()

    def after_val_step(self, runner, epoch_state, step_state):
        log_interval = runner.cfg.log.interval_val
        if self.every_n_inner_steps(epoch_state, log_interval):
            dataset_name = epoch_state.get('dataset_name', '')
            progress = f'Validation {dataset_name} [{epoch_state.inner_step+1}/{epoch_state.val_steps}]'
            metrics_str = self.get_metrics_str(runner.state.log_metrics, epoch_state)
            log_str = f'{progress} {metrics_str}'
    
            logger.info(log_str)

            runner.state.log_metrics.reset()
            self.last_step = epoch_state.inner_step
            self.last_step_timestamp = time.time_ns()
    
    def after_train_epoch(self, runner, epoch_state):
        time_elapsed = (time.time_ns() - self.epoch_start_timestamp) / 1e9
        if epoch_state.inner_step:
            avg_step_time = time_elapsed / epoch_state.inner_step
        else:
            avg_step_time = 0
        time_elapsed_str = timedelta_format(datetime.timedelta(seconds=time_elapsed))
        sep_str = '*********'
        log_str = (f'{sep_str} Finish train epoch {runner.epoch+1}, '
                   f'step time: {avg_step_time:.3f} s, '
                   f'total time: {time_elapsed_str} {sep_str}')
        logger.info(log_str)

    def after_val_epoch(self, runner, epoch_state):
        time_elapsed = (time.time_ns() - self.epoch_start_timestamp) / 1e9
        avg_step_time = time_elapsed / epoch_state.inner_step
        time_elapsed_str = timedelta_format(datetime.timedelta(seconds=time_elapsed))
        sep_str = '*********'
        log_str = (f'{sep_str} Finish validation, '
                   f'step time: {avg_step_time:.3f} s, '
                   f'total time: {time_elapsed_str} {sep_str}')
        logger.info(log_str)
