import torch
from torch.nn.parallel import DistributedDataParallel as torch_DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import broadcast

from .hook import Hook
from simseg.utils import ENV

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex import amp
except ImportError:
    pass


class DistHook(Hook):
    r"""
    A kind of Hook to execute distributed training.
    """

    def __init__(self, runner):
        self.dist = runner.cfg.dist
        self.fp16 = runner.cfg.dist.fp16

    def before_run(self, runner):
        if self.dist.name == 'apex':
            # Wrapping criterion inside amp will lead to unstable loss, examples include
            # inf val loss occurance under label_smooth CE loss. Therefore we remove
            # criterion of the model when using apex no matter which opt_level it uses.
            assert not hasattr(runner.model, 'criterion') or len(list(runner.model.criterion.parameters())) == 0, \
                f'No learnable parameters are allowed in model.criterion when using apex (amp). ' \
                f'You are encouraged to move all the leanable parameters to model.backbone/neck/head. ' \
                f'If you have to include learnable parameters in model.criterion, please set dist.name=torch instead.'

            modules = runner.model.__dict__.get('_modules')
            pipeline_modules = {name: module for name, module in modules.items() if name != 'criterion'}
            pipeline_amp, runner.optimizer = amp.initialize(
                list(pipeline_modules.values()), runner.optimizer,
                opt_level=self.dist.param.get('opt_level', 'O0'))
            for idx, name in enumerate(pipeline_modules.keys()):
                modules[name] = pipeline_amp[idx]
            del pipeline_modules
            del pipeline_amp

            # Regular DistributedDataParallel for the whole model
            runner.model = DDP(runner.model, delay_allreduce=False)
        elif self.dist.name == 'torch':
            runner.model = torch_DDP(runner.model,
                                     device_ids=[ENV.local_rank],
                                     output_device=ENV.local_rank,
                                     find_unused_parameters=False)

            if self.fp16:
                runner.scaler = torch.cuda.amp.GradScaler()

    def before_val_epoch(self, runner, epoch_state):
        # Explicitly synchronize buffers across all devices.
        # Necessary or the validation result will be different from inference
        if self.dist.name in ['apex', 'torch'] and (not runner.cfg.data.get('single_eval')):
            buffers = list(runner.model.buffers())
            for buffer in buffers:
                broadcast(buffer, 0)

    def before_train_epoch(self, runner, epoch_state):
        if hasattr(epoch_state.data_loader, 'sampler') and \
                isinstance(epoch_state.data_loader.sampler, DistributedSampler):
            epoch_state.data_loader.sampler.set_epoch(runner.epoch)
