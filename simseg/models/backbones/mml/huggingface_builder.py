import torch.nn as nn
from transformers import AutoModel, AutoConfig

from ..builder import BACKBONE

class HuggingFaceModel(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(HuggingFaceModel, self).__init__()
        if cfg.model.text_encoder.pretrained:
            self.model = AutoModel.from_pretrained(cfg.model.text_encoder.tag,
                        add_pooling_layer=False)
        else:
            config = AutoConfig.from_pretrained(cfg.model.text_encoder.tag)
            self.model = AutoModel.from_config(config)

    def forward(self, input_ids, attention_mask, **kwargs):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)
        

@BACKBONE.register_obj
def huggingface_modelzoo(cfg, **kwargs):
    model = HuggingFaceModel(cfg, **kwargs)
    return model
