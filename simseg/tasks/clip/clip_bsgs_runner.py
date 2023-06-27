import time
import torch
torch.set_printoptions(precision=7)

from addict import Dict as adict

from torch.nn import functional as F

from simseg.core import DistHook, HookMode, WandbHook
from simseg.core.hooks.log import LogHook
from simseg.core.runners.builder import RUNNER
from simseg.core.runners.epoch_runner import EpochRunner

from simseg.utils.dist import generate_local_groups

from simseg.tasks.clip.hooks import *

from simseg.utils.misc import calc_topk_accuracy
from simseg.utils import all_gather_group, logger, ENV

try:
    from apex import amp
except ImportError:
    pass

import numpy as np
import random

def setup_seed(seed):
    # for stable decoupled gradient accumulation~(DGA).
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

@RUNNER.register_obj
class CLIP_BSGS_Runner(EpochRunner):
    """ A runner used for clip

    Args:
        cfg (adict): global config.
    """

    def __init__(self, cfg, data_loaders, model):
        logger.info("CLIP runner initiated")
        super(CLIP_BSGS_Runner, self).__init__(cfg, data_loaders, model)
        self._init_clip_runner()

        if data_loaders['train_dataset']:
            num_samples = [len(dataset) for dataset in data_loaders['train_dataset']]
            total_num = sum(num_samples)
            self.sample_weights = [num_sample / total_num for num_sample in num_samples]
        else:
            self.sample_weights = None

    def _init_clip_runner(self):
        self.train_type = self.cfg.data.train_type
        self.total_steps = self.train_steps * self.max_epochs
        self.warmup_steps = int(self.total_steps * self.cfg.optim.lr.warmup_proportion)

        self.dist_name = self.cfg.dist.name
        self.fp16 = self.cfg.dist.fp16

        self.batch_size_train = self.cfg.data.batch_size_train // ENV.size
        self.batch_size_val = self.cfg.data.batch_size_val // ENV.size
        self.batch_size = self.cfg.data.batch_size // ENV.size
        assert self.batch_size_val % self.batch_size_train == 0
        assert self.batch_size % self.batch_size_val == 0

        group_size = self.cfg.loss.group_size
        if group_size < 0:
            group_size = ENV.size
        group, group_rank = generate_local_groups(group_size)

        self.rank = group_rank
        self.group = group

        if self.cfg.runner.stable_random != "none":
            assert self.cfg.data.batch_size_train == self.cfg.data.batch_size_val

        # set random seed for sampling from the same dataset.
        self.rng = np.random.default_rng(2021)

    def init_hook(self):
        self.register_hook(ClipOptimizerHook(self),
                            priority='very_high', hook_mode=HookMode.TRAIN)

        self.register_hook(DistHook(self),
                           priority='very_high', hook_mode=HookMode.TRAIN)
        
        self.register_hook(ClipCheckpointHook(self),
                           priority='low', hook_mode=HookMode.TRAIN)
        
        self.register_hook(LogHook(self),
                           priority='very_low')
        
        if self.cfg.data.single_eval:
            self.register_hook(RetrievalLocalEvalHook(self),
                            priority='very_low', hook_mode=HookMode.TRAIN)
        else:
            self.register_hook(RetrievalEvalHook(self),
                            priority='very_low', hook_mode=HookMode.TRAIN)         
        
        if self.cfg.wandb.enable:
            self.register_hook(WandbHook(self),
                            priority='lowest', hook_mode=HookMode.TRAIN)  


    def input_preprocess(self, batch, mode='train'):
        batch = {k: v.cuda(ENV.device, non_blocking=True) for k,v in batch.items() if k not in ['caption', 'name']}
        return batch

    def create_batch_dict(self, batch, mode='train'):
        batch_dict = adict()
        if mode == 'train':
            batch_dict['image'], batch_dict['input_ids'], batch_dict['attention_mask'], \
                batch_dict['caption'] = batch  
        else:
            batch_dict['image'], batch_dict['input_ids'], batch_dict['attention_mask'], \
                batch_dict['caption'], batch_dict['image_id'], batch_dict['caption_id'] = batch  
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

    def sequential_train(self, data_iters, epoch_state, train_steps=None):
        if data_iters is None:
            return
        self.model.train()
        
        self.call_hook('_before_train_epoch', epoch_state)
        for data_iter in data_iters:
            for batch in data_iter:
                step_state = adict()
                batch = self.create_batch_dict(batch)
                batch = self.input_preprocess(batch)
                
                if train_steps and epoch_state.inner_step > train_steps:
                    logger.emph('breaked??')
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
    
    def debias_train(self, data_loaders, epoch_state, train_steps=None):
        if data_loaders is None:
            return
        self.model.train()

        data_iters = [iter(data_loader) for data_loader in data_loaders]
        num_datasets = len(data_loaders)
        
        self.call_hook('_before_train_epoch', epoch_state)


        for i in range(train_steps):
            iter_index = self.rng.choice(num_datasets, p=self.sample_weights)
            try:
                data_iter = data_iters[iter_index]
                batch = next(data_iter)
            except StopIteration:
                data_iters[iter_index] = iter(data_loaders[iter_index])
                data_iter = data_iters[iter_index]
                batch = next(data_iter)
            
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
                step_state.batch_output = self.batch_processor(batch, embeddings=True)
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

            if self.train_type == 'shuffle':
                self.train(train_dataloader, epoch_state, self.train_steps)
            elif self.train_type == 'sequential':
                self.sequential_train(train_dataloader, epoch_state, self.train_steps)
            elif self.train_type == 'debias':
                self.debias_train(train_dataloader, epoch_state, self.train_steps)
            else:
                raise NotImplementedError
            self.epoch += 1

            if self.epoch % self.val_interval == 0 and val_dataloader_list and self.val_interval_steps < 0:
                for val_dataloader, val_steps in zip(val_dataloader_list, val_steps_list):
                    try:
                        val_data_iter = val_dataloader.get_iterator(0, 0)
                    except:
                        val_data_iter = val_dataloader   
                    self.val(val_data_iter, val_steps)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')


    def batch_processor(self, data_batch, embeddings=False):
        if self.cfg.runner.stable_random != "none":
            stable_random_seed = self.step
            setup_seed(stable_random_seed)

        mixup_kwargs = {}
        if self.model.module.use_mixup and not embeddings:
            mixup_kwargs = self.model.module.get_mixup_kwargs(mixup_kwargs)

        with torch.no_grad():
            if embeddings:
                _image_embeddings, _text_embeddings, temp = self.model(data_batch, embeddings='all', **mixup_kwargs)

                output = {'image_embeddings': _image_embeddings,
                        'text_embeddings': _text_embeddings,
                        'image_id': data_batch['image_id'],
                        'caption_id': data_batch['caption_id']
                        }
                return output

            image_embeddings_local, text_embeddings_local = [], []

            for _idx_l in range(0, self.batch_size, self.batch_size_train):
                _data_batch = {"image": data_batch["image"][_idx_l: _idx_l + self.batch_size_train],
                            "input_ids": data_batch["input_ids"][_idx_l: _idx_l + self.batch_size_train],
                            "attention_mask": data_batch["attention_mask"][_idx_l: _idx_l + self.batch_size_train]
                }
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        # (i', d), (t', d)
                        _image_embeddings, _text_embeddings, temp = self.model(_data_batch, embeddings='all', **mixup_kwargs)
                else:
                    # (i', d), (t', d)
                    _image_embeddings, _text_embeddings, temp = self.model(_data_batch, embeddings='all', **mixup_kwargs)

                image_embeddings_local.append(_image_embeddings)
                text_embeddings_local.append(_text_embeddings)
            
            # (i, d), (t, d)
            image_embeddings_local = torch.cat(image_embeddings_local, dim = 0)
            text_embeddings_local = torch.cat(text_embeddings_local, dim = 0)
            
            temp_sqrt = torch.sqrt(temp)

            # (i, d)
            image_embeddings_global = torch.cat(all_gather_group(image_embeddings_local, self.group), 0)
            # (t, d)
            text_embeddings_global = torch.cat(all_gather_group(text_embeddings_local, self.group), 0)
    
            s_i2t_nm = image_embeddings_global @ text_embeddings_local.T
            s_i2t_mn = image_embeddings_local @ text_embeddings_global.T

            # (i, t'), (i', t)
            s_i2t_nm /= temp
            s_i2t_mn /= temp
            
            # (i), (t)
            targets_i2t = torch.arange(self.batch_size * ENV.rank, self.batch_size * (ENV.rank + 1), device = ENV.device)
            targets_t2i = torch.arange(self.batch_size * ENV.rank, self.batch_size * (ENV.rank + 1), device = ENV.device)

            loss = 0.5 * (F.cross_entropy(s_i2t_mn, targets_i2t) + F.cross_entropy(s_i2t_nm.T, targets_t2i)).cpu().item()
            
            y_i2t = torch.eye(self.cfg.data.batch_size, device=image_embeddings_local.device)
            if self.model.module.use_mixup and not embeddings:
                y_i2t_flip = torch.block_diag(*[torch.eye(self.batch_size_train).flip(0)] * (self.cfg.data.batch_size // self.batch_size_train)).to(device=image_embeddings_local.device)
                alpha = mixup_kwargs['image_alpha'] if 'image_alpha' in mixup_kwargs else mixup_kwargs['text_alpha']
                y_i2t = alpha * y_i2t + (1-alpha) * y_i2t_flip
            y_i2t = y_i2t[self.batch_size * ENV.rank: self.batch_size * (ENV.rank + 1), :]

            # (i'), (t')
            s_i2t_esum_local = torch.sum(torch.exp(s_i2t_mn), dim = 1)
            s_t2i_esum_local = torch.sum(torch.exp(s_i2t_nm.T), dim = 1)
            
            # (i), (t)
            s_i2t_esum = torch.cat(all_gather_group(s_i2t_esum_local, self.group), 0).unsqueeze(dim = 1)
            s_t2i_esum = torch.cat(all_gather_group(s_t2i_esum_local, self.group), 0).unsqueeze(dim = 1)

            p_i2t_mn = torch.exp(s_i2t_mn) / s_i2t_esum[self.batch_size * ENV.rank: self.batch_size * (ENV.rank + 1), :]
            p_t2i_nm = torch.exp(s_i2t_mn.T) / s_t2i_esum
            left_I = (p_i2t_mn + p_t2i_nm.T - 2 * y_i2t) @ text_embeddings_global
            
            p_i2t_nm = torch.exp(s_i2t_nm) / s_i2t_esum
            p_t2i_mn = torch.exp(s_i2t_nm.T) / s_t2i_esum[self.batch_size * ENV.rank: self.batch_size * (ENV.rank + 1), :]
            left_T = (p_i2t_nm.T + p_t2i_mn - 2 * y_i2t) @ image_embeddings_global
            
            # (i, d) = (1) * ((i, t) @ (t, d))
            left_I /= temp_sqrt
            left_T /= temp_sqrt

            i2t_acc = calc_topk_accuracy(p_i2t_mn, targets_i2t)[0]  # (1)
            t2i_acc = calc_topk_accuracy(p_t2i_mn, targets_t2i)[0]  # (1)
        
        if self.cfg.runner.stable_random != "none":
            setup_seed(stable_random_seed)

        for _idx_l in range(0, self.batch_size, self.batch_size_train):
            _data_batch = {"image": data_batch["image"][_idx_l: _idx_l + self.batch_size_train],
                        "input_ids": data_batch["input_ids"][_idx_l: _idx_l + self.batch_size_train],
                        "attention_mask": data_batch["attention_mask"][_idx_l: _idx_l + self.batch_size_train]
            }

            # (i', d), (t', d)
            _left_I = left_I[_idx_l: _idx_l + self.batch_size_train]
            _left_T = left_T[_idx_l: _idx_l + self.batch_size_train]

            if self.scaler:
                with torch.cuda.amp.autocast():
                    # (i', d), (t', d)
                    _image_embeddings, _text_embeddings, temp = self.model(_data_batch, embeddings='all', **mixup_kwargs)
            else:
                # (i', d), (t', d)
                _image_embeddings, _text_embeddings, temp = self.model(_data_batch, embeddings='all', **mixup_kwargs)
            
            temp_sqrt = torch.sqrt(temp)

            # (i')
            loss_temp_i = _left_I * _image_embeddings
            loss_temp_t = _left_T * _text_embeddings

            loss_temp = (loss_temp_i + loss_temp_t).sum() / 2 / self.batch_size

            loss_temp = loss_temp / temp_sqrt
            
            if self.dist_name == 'apex':
                with amp.scale_loss(loss_temp, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            elif self.dist_name == 'torch' and self.fp16:
                self.scaler.scale(loss_temp).backward()
            else:
                loss_temp.backward()
        
        output = {'loss': loss,
                'temperature': temp,
                'i2t_acc': i2t_acc,
                't2i_acc': t2i_acc,
                'lr': self.optimizer.param_groups[0]['lr']
                }
        
        self.state.log_metrics.add_store('i2t_acc', i2t_acc)
        self.state.log_metrics.add_store('t2i_acc', t2i_acc)
        self.state.log_metrics.add_store('loss', loss)

        return output

