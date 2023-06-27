import numpy as np
import torch.nn as nn

from simseg.models.pipelines.builder import PIPELINE
from simseg.models.backbones.builder import BACKBONE
from simseg.models.criteria.losses.builder import LOSS

from simseg.utils import logger, ENV
from simseg.tasks.linear_prob.hooks.utils import accuracy

class LinearProbModel(nn.Module):
    def __init__(self, cfg, rank):
        super().__init__()
        self.cfg = cfg
        self.image_encoder = ImageEncoder(cfg)

        assert cfg.model.classifier.num_classes > 0
        self.image_pool = nn.Identity()

        self.classifier = nn.Linear(cfg.model.image_encoder.embedding_dim, cfg.model.classifier.num_classes)


        self.loss = nn.CrossEntropyLoss(reduction='mean')
        
        self.extra_loss_names = cfg.loss.extra_losses
        self.extra_loss = nn.ModuleList()
        if len(cfg.loss.extra_losses) > 0:
            logger.info(f'Using extra losses {cfg.loss.extra_losses}')
            for loss_name in cfg.loss.extra_losses:
                self.extra_loss.append(LOSS.get(loss_name)(cfg, rank))


    def train(self, mode=True):
        # Override train so that the training mode is set as we want (BN does not update the running stats)
        nn.Module.train(self, mode)
        if mode and self.cfg.model.freeze_cnn_bn:
            # fix all bn layers
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find("BatchNorm") != -1:
                    m.eval()

            print("freezing bn in image encoder")
            self.image_encoder.apply(set_bn_eval)

    def forward_image_feature(self, image):
        image_features = self.image_pool(self.image_encoder(image))
        
        # check whether use ViT with only [CLS] token
        if self.cfg.model.image_encoder.vit.only_cls_token and len(image_features.shape) == 3:
            image_features = image_features[:, 0]

        return image_features


    def forward(self, batch, valid=False, **kwargs):

        image_embeddings = self.forward_image_feature(batch["image"])
        prediction = self.classifier(image_embeddings)
        
        loss = self.loss(prediction, batch['label'])
        
        loss_dict = {}
        loss_dict[f'{self.cfg.loss.name}_loss'.lower()] = loss

        if valid:
            return loss_dict, prediction, batch['label']
        else:
            acc1, acc5 = accuracy(prediction, batch['label'], topk=(1, 5))
            return loss_dict, acc1, acc5


class ImageEncoder(nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.cfg = cfg
        self.model_tag = cfg.model.image_encoder.tag
        self.pretrained = cfg.model.image_encoder.pretrained
        self.trainable = cfg.model.image_encoder.trainable

        kwargs_dict = {}
        if "vit" not in self.model_tag: # enable/disable the global average pooling for CNN.
            kwargs_dict["global_pool"] = "avg"
        if "vit" in self.model_tag: # specify the input_size for intializing ViTs with timm.
            kwargs_dict['img_size'] = cfg.transforms.input_size
        
        model_builder = BACKBONE.get(cfg.model.image_encoder.name)
        self.model = model_builder(cfg, **kwargs_dict)

        for p in self.model.parameters():
            p.requires_grad = self.trainable

    def forward(self, x, **kwargs):
        x = self.model(x, **kwargs)
        return x


@PIPELINE.register_obj
def linear_prob(cfg):
    model = LinearProbModel(cfg, ENV.rank)
    return model
