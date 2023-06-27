import torch
import torch.nn as nn
import torch.nn.functional as F

from simseg.utils import all_gather, all_gather_group, logger, ENV, GatherLayer
from simseg.utils.misc import calc_topk_accuracy
from simseg.utils.dist import generate_local_groups

from .builder import LOSS


@LOSS.register_obj
class NCE(nn.Module):
    def __init__(
        self,
        cfg,
        rank,
    ):
        super().__init__()
        self.cfg = cfg
        self.global_reduce = self.cfg.loss.global_reduce

        if self.global_reduce:
            group_size = cfg.loss.group_size
            if group_size < 0:
                group_size = ENV.size
            group, group_rank = generate_local_groups(group_size)

            self.rank = group_rank
            self.group = group
            self.gather_backward = cfg.loss.nce_loss.gather_backward
            
            logger.info('NCE Loss Group size, Group Rank, Env Rank:', group_size, self.rank, ENV.rank, root_only=False)
            if self.gather_backward:
                logger.info('NCE Loss gather grad will backward')

        if self.cfg.loss.temperature.name == 'constant':
            self.temperature = torch.ones([]) * cfg.loss.temperature.value
        elif self.cfg.loss.temperature.name == 'parameter':
            self.temperature = torch.nn.Parameter(torch.ones([]) * cfg.loss.temperature.value) 
        else:
            raise NotImplementedError

        if cfg.loss.smoothing > 0:
            self.cross_entropy = LabelSmoothingCrossEntropy(cfg=self.cfg, rank=self.rank,
                smoothing=cfg.loss.smoothing, reduction="none"
            )
        else:
            self.cross_entropy = nn.CrossEntropyLoss(reduction="none")

    def forward(self, feat1, feat2, label=None, ignore_mask=None):
        N1 = feat1.size(0)
        if ignore_mask is None:
            ignore_mask = torch.zeros(N1, device=feat1.device)

        temp = torch.clamp(self.temperature, 0.001, 0.5)
        
        if self.global_reduce:
            if self.gather_backward:
                feat2_global = GatherLayer.apply(feat2, self.group, self.rank)
                ignore_mask_global = GatherLayer.apply(ignore_mask, self.group, self.rank)
            else:
                feat2_global = torch.cat(all_gather_group(feat2, self.group), 0)
                ignore_mask_global = torch.cat(all_gather_group(ignore_mask, self.group), 0)
                
            N2 = feat2_global.size(0)

            assert N2 % N1 == 0, "global size: {}, batch size: {}".format(N2, N1)

            ignore_mask_global = torch.unsqueeze(ignore_mask_global, 1)
            feat2_global = feat2_global * (1 - ignore_mask_global)
            
            logits = (feat1 @ feat2_global.T) / temp

            targets = torch.arange(N1 * self.rank, N1 * (self.rank + 1), device=feat1.device)

            loss = self.cross_entropy(logits, targets)

        else:
            ignore_mask = torch.unsqueeze(ignore_mask, 1)
            feat2 = feat2 * (1 - ignore_mask)

            logits = (feat1 @ feat2.T) / temp
            targets = torch.arange(N1, device=logits.device)


            loss = 0.5 * (self.cross_entropy(logits, targets) + self.cross_entropy(logits.T, targets))

        loss = loss * (1 - ignore_mask)
        loss = loss.squeeze()
        loss = loss.mean()

        index = ignore_mask < 1
        if self.global_reduce:
            nce_acc = calc_topk_accuracy(logits[index], targets[index])[0]
            return loss, nce_acc
        else:
            if len(index.size()) > 1:
                index = index.squeeze(1)
            i2t_acc = calc_topk_accuracy(logits[index], targets[index])[0]
            t2i_acc = calc_topk_accuracy(logits.T[index], targets[index])[0]

            return loss, i2t_acc, t2i_acc

@LOSS.register_obj
class MixUpNCE(nn.Module):
    def __init__(
        self,
        cfg,
        rank,
    ):
        super().__init__()
        self.cfg = cfg
        self.global_reduce = self.cfg.loss.global_reduce

        if self.global_reduce:
            group_size = cfg.loss.group_size
            if group_size < 0:
                group_size = ENV.size
            group, group_rank = generate_local_groups(group_size)

            self.rank = group_rank
            self.group = group
            self.gather_backward = cfg.loss.nce_loss.gather_backward
            
            logger.info('NCE Loss Group size, Group Rank, Env Rank:', group_size, self.rank, ENV.rank, root_only=False)
            if self.gather_backward:
                logger.info('NCE Loss gather grad will backward')
        else:
            raise NotImplementedError

        if self.cfg.loss.temperature.name == 'constant':
            self.temperature = torch.ones([]) * cfg.loss.temperature.value
        elif self.cfg.loss.temperature.name == 'parameter':
            self.temperature = torch.nn.Parameter(torch.ones([]) * cfg.loss.temperature.value) 
        else:
            raise NotImplementedError

        if cfg.loss.smoothing > 0:
            self.cross_entropy = LabelSmoothingCrossEntropy(cfg=self.cfg, rank=self.rank,
                smoothing=cfg.loss.smoothing, reduction="none"
            )
        else:
            self.cross_entropy = nn.CrossEntropyLoss(reduction="none")

    def forward(self, feat1, feat2, label=None, ignore_mask=None, feat_prior=None, **kwargs):
        if ('image_alpha' in kwargs) and ('text_alpha' not in kwargs):
            alpha = kwargs['image_alpha']
        elif ('text_alpha' in kwargs) and ('image_alpha' not in kwargs):
            alpha = kwargs['text_alpha']
        else:
            raise NotImplementedError('MixUpNCE loss only supports mixing up one modal, please consider M3ixupNCE')

        N1 = feat1.size(0)
        if ignore_mask is None:
            ignore_mask = torch.zeros(N1, device=feat1.device)

        temp = torch.clamp(self.temperature, 0.001, 0.5)
        
        if self.global_reduce:
            if self.gather_backward:
                feat2_global = GatherLayer.apply(feat2, self.group, self.rank)
                ignore_mask_global = GatherLayer.apply(ignore_mask, self.group, self.rank)
            else:
                feat2_global = torch.cat(all_gather_group(feat2, self.group), 0)
                ignore_mask_global = torch.cat(all_gather_group(ignore_mask, self.group), 0)
                
            N2 = feat2_global.size(0)

            assert N2 % N1 == 0, "global size: {}, batch size: {}".format(N2, N1)

            ignore_mask_global = torch.unsqueeze(ignore_mask_global, 1)
            feat2_global = feat2_global * (1 - ignore_mask_global)
            
            logits = (feat1 @ feat2_global.T) / temp

            targets = torch.arange(N1 * self.rank, N1 * (self.rank + 1), device=feat1.device)
            flip_targets = targets.flip(0)
            loss = alpha * self.cross_entropy(logits, targets) + (1 - alpha) * self.cross_entropy(logits, flip_targets)

        else:
            raise NotImplementedError

        loss = loss * (1 - ignore_mask)
        loss = loss.squeeze()
        loss = loss.mean()

        index = ignore_mask < 1
        if self.global_reduce:
            nce_acc = calc_topk_accuracy(logits[index], targets[index])[0]
            return loss, nce_acc
        else:
            if len(index.size()) > 1:
                index = index.squeeze(1)
            i2t_acc = calc_topk_accuracy(logits[index], targets[index])[0]
            t2i_acc = calc_topk_accuracy(logits.T[index], targets[index])[0]
            return loss, i2t_acc, t2i_acc


@LOSS.register_obj
class MSE(nn.Module):
    def __init__(
        self,
        cfg,
        rank,
    ):
        super().__init__()
        group_size = cfg.loss.group_size
        if group_size > 0:
            group, group_rank = generate_local_groups(group_size)
            self.rank = group_rank
            self.group = group
            self.gather_func = all_gather_group
        else:
            self.rank = rank
            self.group = str(rank)
            self.gather_func = all_gather

        self.rank = rank
        self.cfg = cfg
        self.mse_loss = nn.MSELoss()
        self.global_reduce = self.cfg.loss.global_reduce

    def forward(self, feat1_sim, feat2, feat1, label=None, ignore_mask=None):
        N1 = feat1.size(0)

        if ignore_mask is None:
            ignore_mask = torch.zeros(N1, device=feat1.device)

        if self.global_reduce:
            with torch.no_grad():
                feat2_global = torch.cat(self.gather_func(feat2, self.group), 0)
                N2 = feat2_global.size(0)

                assert N2 % N1 == 0, "global size: {}, batch size: {}".format(N2, N1)

                logits = feat1 @ feat2_global.T
                targets = torch.arange(N1, device=logits.device)
                targets += N1 * self.rank
            loss = self.mse_loss(feat1_sim, feat2)

        else:
            with torch.no_grad():
                logits = feat1 @ feat2.T
                targets = torch.arange(N1, device=logits.device)
            loss = self.mse_loss(feat1_sim, feat2)

        loss = loss * (1 - ignore_mask)
        loss = loss.mean()

        index = ignore_mask < 1
        nce_acc = calc_topk_accuracy(logits[index], targets[index])[0]
        return loss, nce_acc


@LOSS.register_obj
class Triplet(nn.Module):
    def __init__(self, cfg, rank):
        super().__init__()
        group_size = cfg.loss.group_size
        if group_size > 0:
            group, group_rank = generate_local_groups(group_size)
            self.rank = group_rank
            self.group = group
            self.gather_func = all_gather_group
        else:
            self.rank = rank
            self.group = str(rank)
            self.gather_func = all_gather

        self.cfg = cfg
        self.rank = rank
        self.ohem = self.cfg.loss.ohem
        self.margin = cfg.loss.triplet_loss.margin
        self.reduce = (
            cfg.loss.triplet_loss.reduce_mode
        )  # "max" triggers the hard negative mining strategy
        self.global_reduce = cfg.loss.global_reduce

    def forward(self, feat1, feat2, label=None, ignore_mask=None):
        N1 = feat1.size(0)
        if ignore_mask is None:
            ignore_mask = torch.zeros(N1, device=feat1.device)

        if self.global_reduce:
            feat2_global = torch.cat(self.gather_func(feat2, self.group), 0)
            # compute cross-modal score matrix
            scores = feat1.mm(feat2_global.t())
            diagonal = (
                scores[:, N1 * self.rank : N1 * (self.rank + 1)].diag().view(N1, 1)
            )
            d1 = diagonal.expand_as(scores)
            d2 = diagonal.t().expand_as(scores.t())

            targets = torch.arange(N1, device=scores.device)
            targets += N1 * self.rank

            loss = (self.margin + scores - d1).clamp(min=0)

            mask = torch.zeros_like(loss, device=scores.device)
            mask[:, N1 * self.rank : N1 * (self.rank + 1)] += 1
            loss = loss.masked_fill_(mask.bool(), 0)

            if self.reduce == "mean":
                loss = loss.sum(1) / (N1 - 1)
            elif self.reduce == "max":
                loss = loss.max(1)[0]
            else:
                raise NotImplementedError(
                    "Reduce method {} is not " "implemented yet".format(self.reduce)
                )

            nce_acc = calc_topk_accuracy(scores, targets)[0]
            return loss.sum(), nce_acc

        else:
            scores = feat1.mm(feat2.t())
            diagonal = scores.diag().view(N1, 1)
            d1 = diagonal.expand_as(scores)
            d2 = diagonal.t().expand_as(scores)
            targets = torch.arange(N1, device=scores.device)

            loss_1to2 = (self.margin + scores - d1).clamp(min=0)
            loss_2to1 = (self.margin + scores - d2).clamp(min=0)

            # clear diagonals
            mask = torch.eye(N1, device=scores.device) > 0.5
            loss_1to2 = loss_1to2.masked_fill_(mask, 0)
            loss_2to1 = loss_2to1.masked_fill_(mask, 0)

            # keep the maximum violating negative for each query
            if self.reduce == "mean":
                loss_1to2 = loss_1to2.sum(1) / (N1 - 1)
                loss_2to1 = loss_2to1.sum(0) / (N1 - 1)
            elif self.reduce == "max":
                loss_1to2 = loss_1to2.max(1)[0]
                loss_2to1 = loss_2to1.max(0)[0]
            else:
                raise NotImplementedError(
                    "Reduce method {} is not " "implemented yet".format(self.reduce)
                )
            loss = loss_1to2 + loss_2to1

            i2t_acc = calc_topk_accuracy(scores, targets)[0]
            t2i_acc = calc_topk_accuracy(scores.T, targets)[0]

            return loss.sum(), i2t_acc, t2i_acc


@LOSS.register_obj
class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, cfg, rank, smoothing=0.1, reduction="mean", **kwargs):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.reduction = reduction

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        if self.reduction == "mean":
            return loss.mean()
        else:
            return loss


@LOSS.register_obj
class SoftTargetCrossEntropy(nn.Module):
    def __init__(self, cfg, rank, reduction="mean", **kwargs):
        super(SoftTargetCrossEntropy, self).__init__()
        self.reduction = reduction

    def forward(self, x, target):
        loss = -target * F.log_softmax(x, dim=-1)
        loss = torch.sum(loss, dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        else:
            return loss
