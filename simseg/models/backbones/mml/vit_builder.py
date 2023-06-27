import timm
import torch
import torch.nn as nn

from ..builder import BACKBONE


class ViTModel(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(ViTModel, self).__init__()
        self.model = timm.create_model(cfg.model.image_encoder.tag, pretrained=cfg.model.image_encoder.pretrained, num_classes=0, **kwargs)

    def forward(self, x):
        x = self.model.patch_embed(x)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.model.pos_drop(x + self.model.pos_embed)
        x = self.model.blocks(x)
        x = self.model.norm(x)
        x = torch.reshape(x, (x.size(0), x.size(1), -1))
        return x


@BACKBONE.register_obj
def vit_modelzoo(cfg, **kwargs):
    model = ViTModel(cfg, **kwargs)
    return model
