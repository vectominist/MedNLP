'''
    Implementation of 
    DCMN+: Dual Co-Matching Network for Multi-Choice Reading Comprehension
'''
import torch
from torch import nn
import torch.nn.functional as F


class SingleMatching(nn.Module):
    '''
        Single Matching Network
    '''

    def __init__(self, dim: int):
        super(SingleMatching, self).__init__()

        self.

    def forward(
            self,
            h_a: torch.Tensor,
            h_b: torch.Tensor):
        '''
            h_a: B x D
            h_b: B x D
        '''



class DCMN(nn.Module):
    '''
        Main network for DCMN+
        This is a simplified implementation.
    '''

    def __init__(self, dim: int, topk: int):
        super(DCMN, self).__init__()

        self.dim = dim
        self.topk = topk
        # self.choice_compare = nn.Linear(dim, dim, bias=False)
        self.bi_matching = BidirectionalMatching(dim, dim * self.topk)

    def forward(
            self,
            doc_embs: torch.Tensor,
            stem_embs: torch.Tensor,
            choice_embs: torch.Tensor,
            doc_len: torch.Tensor):
        '''
            doc_embs: B x S x D
            stem_embs: B x D
            choice_embs: B x 3 x D
            doc_len: B
        '''

        B, S, D = doc_embs.shape

        # Passage sentence selection
        doc_embs_match = doc_embs.unsqueeze(2).expand(-1, -1, 3, -1)
        choice_embs_match = choice_embs.unsqueeze(1).expand(
            -1, doc_embs.shape[1], -1, -1)
        dist_doc_choice = F.cosine_similarity(
            doc_embs_match, choice_embs_match, dim=3)  # B x S x 3

        dist_doc_stem = F.cosine_similarity(
            doc_embs,
            stem_embs.unsqueeze(1).expand(-1, doc_embs.shape[1], -1),
            dim=3)  # B x S

        scores = dist_doc_choice.mean(2) + dist_doc_stem  # B x S

        mask = torch.zeros((B, S), dtype=torch.bool, device=doc_embs.device)
        for b in range(B):
            mask[b, doc_len[b]:] = True
        scores.masked_fill_(mask, -1e9)

        _, topk_idx = torch.topk(scores, self.topk, dim=1)
        topk_sents = doc_embs[range(B), topk_idx[range(B), :], :]
        # topk_sents: B x K x D

        # TODO: Option interaction
        # G = torch.bmm(
        #     choice_embs, self.choice_compare(choice_embs).transpose(1, 2))
        # G = torch.softmax(G, dim=2)  # B x 3 x 3
        # H_aij = torch.relu()
