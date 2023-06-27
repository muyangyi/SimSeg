import torch
import torch.nn as nn

__all__ = ["TopKPooling", "AvgPooling", "VanillaTopKPooling"]


class AvgPooling(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x, attention_mask=None):
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1) 
            x = (x * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            x = x.mean(dim=1)
        return x


class VanillaTopKPooling(nn.Module):
    def __init__(
        self,
        k,
        dim
    ):
        super().__init__()
        self.k = k
        self.dim = dim
    
    def maxk_pool(self, x, reduce_dim, k):
        """ Performs max-k pooling along a given `reduce_dim` """
        index = x.topk(k, dim=reduce_dim)[1]
        return x.gather(reduce_dim, index)

    def forward(self, x):
        maxk_selected_x = self.maxk_pool(x, self.dim, self.k)
        return maxk_selected_x.mean(self.dim)


class TopKPooling(nn.Module):
    def __init__(
        self,
        k,
        dim
    ):
        super().__init__()
        self.k = k
        self.dim = dim
    
    def maxk_pool(self, x, reduce_dim, k):
        """ Performs max-k pooling along a given `reduce_dim` """
        index = x.topk(k, dim=reduce_dim)[1]
        return x.gather(reduce_dim, index)

    def forward(self, x, attention_mask=None):
        k = self.k
        if attention_mask is not None:
            x[torch.where(attention_mask == 0)] = -10000
            min_length = min(attention_mask.sum(1))
            if min_length < k:
                k = min_length
        maxk_selected_x = self.maxk_pool(x, self.dim, k)
        return maxk_selected_x.mean(self.dim)
