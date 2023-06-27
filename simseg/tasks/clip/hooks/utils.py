import glob
from dataclasses import dataclass
from typing import Any, Dict

import torch


@dataclass
class IndexedEmbInfo:
    emb_name: str
    group_idx: torch.Tensor  # [N]
    emb_mat: torch.Tensor  # [N,D]

    def unique(self):
        gidx, gidx_indices = torch.sort(self.group_idx)
        emb_mat = self.emb_mat[gidx_indices]
        uni_idx, uni_count = torch.unique_consecutive(gidx, return_counts=True)
        uni_offset = torch.cumsum(uni_count, 0) - 1
        return IndexedEmbInfo(self.emb_name, uni_idx, emb_mat[uni_offset])

    def to_chunks(self, chunk_size):
        total = self.emb_mat.shape[0]
        for start in range(0, total, chunk_size):
            yield IndexedEmbInfo(
                self.emb_name,
                self.group_idx[start : start + chunk_size],
                self.emb_mat[start : start + chunk_size],
            )


class EmbANN:
    def __init__(self, chunk_size=None) -> None:
        self.chunk_size = chunk_size

    def _ann(self, leftemb: IndexedEmbInfo, rightemb: IndexedEmbInfo):
        emb_sim = leftemb.emb_mat @ rightemb.emb_mat.T
        left_gid = leftemb.group_idx.unsqueeze(1).expand_as(emb_sim)
        right_gid = rightemb.group_idx.unsqueeze(0).expand_as(emb_sim)
        right_simrank = torch.argsort(emb_sim, dim=1, descending=True)
        rightgid_sorted = torch.gather(right_gid, 1, right_simrank)  # (N,D)
        rightgid_matched = rightgid_sorted == left_gid
        return rightgid_sorted, rightgid_matched

    def __call__(self, leftemb: IndexedEmbInfo, rightemb: IndexedEmbInfo):
        if self.chunk_size is None:
            return self._ann(leftemb, rightemb)
        sub_list = []
        for sub_emb in leftemb.to_chunks(self.chunk_size):
            sub_list.append(self._ann(sub_emb, rightemb))
        return torch.cat(sub_list, dim=0)


class RetrievalMetric:
    def __init__(self, with_prefix=True) -> None:
        self.recall_range = (1, 5, 10)
        self.with_prefix = with_prefix
        self.ann = EmbANN()

    def __call__(self, leftemb: IndexedEmbInfo, rightemb: IndexedEmbInfo):
        """
        left (M,D) <- right (N,D)
        """
        _, rightgid_matched = self.ann(leftemb, rightemb)
        leftgid_hasmatch, leftgid_firstmatch = torch.max(rightgid_matched, dim=1)
        leftmatch_rank = leftgid_firstmatch[leftgid_hasmatch]
        assert leftmatch_rank.shape[0] > 0
        result_dict: Dict[str, Any] = {}
        for recall_bound in self.recall_range:
            match_count = (leftmatch_rank < recall_bound).sum()
            totcal_count = leftgid_hasmatch.sum()
            result_dict[f"R@{recall_bound}"] = (match_count / totcal_count).item()
        if self.with_prefix:
            prefix = f"[{leftemb.emb_name}] to [{rightemb.emb_name}]:"
            result_dict = {f"{prefix} {suffix}": result_dict[suffix] for suffix in result_dict}
        return result_dict
