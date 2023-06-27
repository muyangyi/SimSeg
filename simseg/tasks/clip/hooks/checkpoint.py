#!/usr/bin/env python
import re

import torch

from simseg.core.hooks import CheckpointHook
from simseg.core.hooks.checkpoint import get_dist_state_dict

from simseg.utils import logger, filter_state
from simseg.utils.interpolate_pe import interpolate_pos_embed

class ClipCheckpointHook(CheckpointHook):

    def __init__(self, runner):
        super(ClipCheckpointHook, self).__init__(runner)
        self.cfg = runner.cfg

    def resume_from_external(self, runner):
        # load checkpoint
        cfg = runner.cfg
        if not cfg.ckpt.external_resume:
            return

        model = runner.model

        checkpoint = torch.load(cfg.ckpt.external_resume, map_location="cpu")
        checkpoint = self.preprocess_external_checkpoint(checkpoint, model)
        model_checkpoint = get_dist_state_dict(checkpoint['model'])

        logger.emph(f'=> Loading pretrained model: {cfg.ckpt.external_resume}\n')

        # load model
        checkpoint_renamed, dismatching_keys, missing_keys, unexpected_keys = filter_state(
            model.state_dict(), model_checkpoint, cfg.model.pretrain_prefix_change_list)

        if len(dismatching_keys) > 0:
            logger.warning("************* Keys with dismatched shape *************")
            logger.warning(dismatching_keys)
        if len(missing_keys) > 0:
            logger.warning("*************** Keys missing in checkpoint ***************")
            logger.warning(missing_keys)
        if len(unexpected_keys) > 0:
            logger.warning("************** Unexpected keys in checkpoint *************")
            logger.warning(unexpected_keys)

        if self.cfg.ckpt.only_load_image_encoder:
            model.load_state_dict(checkpoint_renamed, strict=False)
            logger.emph('Loading only image encoder.')
        elif self.cfg.ckpt.only_load_text_encoder:
            model.load_state_dict(checkpoint_renamed, strict=False)
            logger.emph('Loading only text encoder.')
        else:
            model.load_state_dict(checkpoint_renamed, strict=not cfg.ckpt.soft_resume)
            assert len(dismatching_keys + missing_keys) == 0

        logger.emph(f'=> Loaded pretrained model: {cfg.ckpt.external_resume}\n')

    def preprocess_checkpoint(self, checkpoint, model=None):
        if 'model' not in checkpoint:
            logger.emph("Preprocessing legacy model (Removing prefix & change key to state_dict)")

            # model key compatibility
            legacy_model_key = 'model_state_dict'
            if legacy_model_key not in checkpoint:
                legacy_model_key = 'state_dict'
            assert (legacy_model_key in checkpoint), "couldn't find model in checkpoint file!"

            checkpoint_renamed = {}
            checkpoint['model'] = checkpoint_renamed

            for attr in checkpoint[legacy_model_key]:
                new_key = re.sub('^module\.', '', attr)
                checkpoint_renamed[new_key] = checkpoint[legacy_model_key][attr]
            checkpoint.pop(legacy_model_key)

        return checkpoint

    def preprocess_external_checkpoint(self, checkpoint, model=None):
        checkpoint = self.preprocess_checkpoint(checkpoint, model=model)

        if self.cfg.model.interpolate_pos_embed and model is not None:
            pos_embed_reshaped = interpolate_pos_embed(checkpoint['model']['image_encoder.model.model.pos_embed'], model.module.image_encoder.model.model)   
            checkpoint['model']['image_encoder.model.model.pos_embed'] = pos_embed_reshaped
            logger.info('Interpolate PE successed.')

        if self.cfg.ckpt.only_load_image_encoder:
            checkpoint_only_image_encoder = {}
            for attr in checkpoint['model']:
                if 'image' in attr:
                    checkpoint_only_image_encoder[attr] = checkpoint['model'][attr]
            checkpoint['model'] = checkpoint_only_image_encoder

        if self.cfg.ckpt.only_load_text_encoder:
            checkpoint_only_image_encoder = {}
            for attr in checkpoint['model']:
                if 'text' in attr:
                    checkpoint_only_image_encoder[attr] = checkpoint['model'][attr]
            checkpoint['model'] = checkpoint_only_image_encoder


        return checkpoint
