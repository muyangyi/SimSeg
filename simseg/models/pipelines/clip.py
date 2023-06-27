import numpy as np

import torch
import torch.nn as nn

from simseg.models.pipelines.builder import PIPELINE
from simseg.models.backbones.builder import BACKBONE
from simseg.models.criteria.losses.builder import LOSS

from simseg.utils import logger, ENV
from ..components import SimpleProjection, ComplexProjection, AvgPooling, TopKPooling, L2norm

class CLIPModel(nn.Module):
    def __init__(self, cfg, rank):
        super().__init__()
        self.cfg = cfg
        self.image_encoder = ImageEncoder(cfg)
        self.text_encoder = TextEncoder(cfg)
        self.random_seed = np.random.RandomState(seed=2021)

        if cfg.model.projection.name == "simple":
            ProjectionHead = SimpleProjection
        elif cfg.model.projection.name == "complex":
            ProjectionHead = ComplexProjection
        else:
            raise NotImplementedError

        self.image_projection = ProjectionHead(
            cfg, embedding_dim=cfg.model.image_encoder.embedding_dim, projection_dim=cfg.model.projection.dim, trainable=cfg.model.projection.image_projector_trainable
            )
        self.text_projection = ProjectionHead(
            cfg, embedding_dim=cfg.model.text_encoder.embedding_dim, projection_dim=cfg.model.projection.dim, trainable=cfg.model.projection.text_projector_trainable
            )

        if cfg.model.pool.name == "loda":
            self.text_pool = TopKPooling(cfg.model.pool.loda.text_k, dim=1)
            self.image_pool = TopKPooling(cfg.model.pool.loda.image_k, dim=1)
        elif cfg.model.pool.name == "avg":
            self.text_pool = AvgPooling()
            self.image_pool = AvgPooling()
        else:            
            self.text_pool = nn.Identity()
            self.image_pool = nn.Identity()

        self.loss = LOSS.get(cfg.loss.name)(cfg, rank)

        self.global_reduce = cfg.loss.global_reduce
        self.text_target_token_idx = cfg.model.text_encoder.target_token_idx


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
        image_features = self.image_encoder(image)

        # pre-pooling
        if self.cfg.model.pool.name == "identity":
            # cls token
            if len(image_features.shape) == 3:
                image_features = image_features[:, 0]
        else:
            # patch tokens 
            if len(image_features.shape) == 3:
                image_features = image_features[:, 1:]
        
        # reshape only for CNN features (N,C,H,W) -> (N, HW, C)
        if len(image_features.shape) == 4:
            image_features = torch.reshape(
                image_features, (image_features.size(0), image_features.size(1), -1)
            ).transpose(1, 2)

        return image_features


    def forward_image_project(self, image_features):
        image_embeddings = self.image_pool(self.image_projection(image_features))

        if self.cfg.model.projection.name == 'simple':
            image_embeddings = L2norm(image_embeddings, dim=-1)

        return image_embeddings


    def forward_text_feature(self, input_ids, attention_mask):
        text_features = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        # pre-pooling
        if self.cfg.model.pool.name == "identity":
            # cls token
            text_features = text_features[:, self.text_target_token_idx, :].squeeze(dim=1)
        else:
            # word tokens 
            text_features = text_features[:, self.text_target_token_idx:, :]

        return text_features


    def forward_text_project(self, text_features, attention_mask):
        if self.cfg.model.pool.name == "identity":
            text_embeddings = self.text_pool(self.text_projection(text_features))
        else:
            text_embeddings = self.text_pool(self.text_projection(text_features), attention_mask)

        if self.cfg.model.projection.name == 'simple':
            text_embeddings = L2norm(text_embeddings, dim=-1)

        return text_embeddings

    
    def forward_loss(
        self,
        image_embeddings,
        text_embeddings,
        ignore_mask=None,
    ):
        if self.global_reduce:
            i2t_loss, i2t_acc = self.loss(
                image_embeddings,
                text_embeddings,
                ignore_mask=ignore_mask,
            )
            t2i_loss, t2i_acc = self.loss(
                text_embeddings,
                image_embeddings,
                ignore_mask=ignore_mask,
            )
            loss = 0.5 * (i2t_loss + t2i_loss)
        else:
            loss, i2t_acc, t2i_acc = self.loss(
                image_embeddings, text_embeddings, ignore_mask=ignore_mask
            )

        loss_dict = {}
        loss_dict[f"{self.cfg.loss.name}_loss".lower()] = loss
        
        return loss_dict, i2t_acc, t2i_acc


    def forward(self, batch, embeddings=False):
        # Getting Image and Text Features
        if embeddings == "image":
            image_embeddings = self.forward_image_feature(batch["image"])
            return image_embeddings
        elif embeddings == "text":
            text_embeddings = self.forward_text_feature(batch["input_ids"], batch["attention_mask"])
            return text_embeddings

        image_embeddings = self.forward_image_feature(batch["image"])
        text_embeddings = self.forward_text_feature(batch["input_ids"], batch["attention_mask"])

        image_embeddings = self.forward_image_project(image_embeddings)
        text_embeddings = self.forward_text_project(text_embeddings, batch["attention_mask"])

        if embeddings == "all":
            return [image_embeddings, text_embeddings]

        # Calculating the Loss
        loss_dict, i2t_acc, t2i_acc = self.forward_loss(
            image_embeddings,
            text_embeddings,
            ignore_mask=None,
        )
        return loss_dict, i2t_acc, t2i_acc


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
            kwargs_dict["global_pool"] = ""
        if "vit" in self.model_tag: # specify the input_size for intializing ViTs with timm.
            kwargs_dict['img_size'] = cfg.transforms.input_size
        
        model_builder = BACKBONE.get(cfg.model.image_encoder.name)
        self.model = model_builder(cfg, **kwargs_dict)

        for p in self.model.parameters():
            p.requires_grad = self.trainable

    def forward(self, x):
        x = self.model(x)
        return x


class TextEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model_tag = cfg.model.text_encoder.tag
        self.pretrained = cfg.model.text_encoder.pretrained
        self.trainable = cfg.model.text_encoder.trainable

        model_builder = BACKBONE.get(cfg.model.text_encoder.name)
        self.model = model_builder(cfg)

        for p in self.model.parameters():
            p.requires_grad = self.trainable

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state


@PIPELINE.register_obj
def clip(cfg):
    model = CLIPModel(cfg, ENV.rank)
    return model
