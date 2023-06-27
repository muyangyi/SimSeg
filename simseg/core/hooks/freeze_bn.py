from .hook import Hook
from simseg.models.components import normalization
from simseg.utils import ENV, logger


class FreezeBNHook(Hook):
    r"""
    A kind of Hook to freeze BN parameters, including mean, variance, scale and bias.
    """

    def __init__(self, runner):
        self.freeze_bn = runner.cfg.model.param.get('freeze_bn', False)
        self.freeze_bn_affine = runner.cfg.model.param.get('freeze_bn_affine', False)

    def before_train_epoch(self, runner, epoch_state):
        if self.freeze_bn:
            logger.info("=> Freeze BN mean and variance.")
            if self.freeze_bn_affine:
                logger.info("=> Freeze BN scale and bais.")
            normalization.convert_freeze_bn(runner.model, freeze_affine=self.freeze_bn_affine)
