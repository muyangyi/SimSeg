import timm
import torch.nn as nn

from ..builder import BACKBONE


class TimmModel(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(TimmModel, self).__init__()
        self.model = timm.create_model(cfg.model.image_encoder.tag, pretrained=cfg.model.image_encoder.pretrained, num_classes=0, **kwargs)

    def forward(self, x):
        x = self.model(x)
        return x


@BACKBONE.register_obj
def timm_modelzoo(cfg, **kwargs):
    model = TimmModel(cfg, **kwargs)
    return model
