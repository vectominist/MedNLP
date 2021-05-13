import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

'''
Implement the PE function.
'''

class PositionalEncoding(nn.Module):
    def __init__(self, d_emb: int, dropout: float = 0.1, max_len: int = 200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        self.pe = torch.zeros(max_len, d_emb)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_emb, 2) *
                             -(math.log(10000.0) / d_emb))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, src):
        pe = self.pe.detach().to(src.device)
        output = src + pe[:, :src.size(1)]
        return self.dropout(output)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_hid, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_hid // n_head
        self.query = nn.Parameter(torch.randn(1, d_hid))
        self.key = nn.Linear(d_hid, d_hid)
        self.value = nn.Linear(d_hid, d_hid)
        self.linear = nn.Linear(d_hid, d_hid)

    def forward(self, x, batch_tk_mask):
        B = x.size(0)

        # Input  shape: `(1, Hid)`.
        # Output shape: `(Head, 1, K)`.
        q = self.query.view(-1, self.n_head, self.d_k).transpose(0, 1)

        # Transform temporal features to query, key and value features.
        # Input  shape: `(B, S, Hid)`.
        # Output shape: `(B, Head, S, K)`.
        k = self.key(x).view(B, -1, self.n_head, self.d_k).transpose(1, 2)
        v = self.value(x).view(B, -1, self.n_head, self.d_k).transpose(1, 2)

        # Calculate self attention scores with query and key features.
        # Self attention scores are scaled down by hidden dimension square
        # root to avoid overflow.
        # `q` Input  shape: `(Head, 1, K)`.
        # `k` Input  shape: `(B, Head, S, K)`.
        # Output shape: `(B, Head, 1, S)`.
        # print(q.shape)
        # print(k.shape)
        attn = q @ k.transpose(-1, -2) / math.sqrt(x.size(-1))

        # Mask parts of attention scores by replacing with large negative
        # values.
        # Input  shape: `(B, Head, 1, S)`.
        # Output shape: `(B, Head, 1, S)`.
        batch_tk_mask = batch_tk_mask.repeat(self.n_head, 1, 1, 1)

        batch_tk_mask = batch_tk_mask.transpose(0, 1)
        batch_tk_mask = batch_tk_mask.transpose(-1, -2)
        # print(attn.shape)
        # print(batch_tk_mask.shape)
        attn.masked_fill_(batch_tk_mask, -1e9)

        # Softmax normalize on attention scores.
        # Large negative values will be closed to zero after normalization.
        # Input  shape: `(B, Head, 1, S)`.
        # Output shape: `(B, Head, 1, S)`.
        attn = F.softmax(attn, dim=-1)

        # Use attention scores to calculate weighted sum on value features.
        # Then perform one more linear tranformation on weighted sum.
        # Finally dropout transformed features.
        # `attn` Input  shape: `(B, Head, 1, S)`.
        # `v` Input  shape: `(B, Head, S, k)`.
        # Output shape: `(B, Head, 1, K)`.
        output = attn @ v

        # Input  shape: `(B, Head, 1, K)`.
        # Output shape: `(B, 1, Hid)`.
        output = output.transpose(1, 2).contiguous()
        output = output.view(B, -1, self.n_head * self.d_k)

        # Output shape: `(B, Hid)`.
        return self.linear(output.squeeze(1))

    def qa(self, x, batch_tk_mask, question):
        B = x.size(0)

        # Input  shape: `(1, Hid)`.
        # Output shape: `(Head, 1, K)`.
        q = question.view(B, -1, self.n_head, self.d_k).transpose(1, 2)

        # Transform temporal features to query, key and value features.
        # Input  shape: `(B, S, Hid)`.
        # Output shape: `(B, Head, S, K)`.
        k = self.key(x).view(B, -1, self.n_head, self.d_k).transpose(1, 2)
        v = self.value(x).view(B, -1, self.n_head, self.d_k).transpose(1, 2)

        # Calculate self attention scores with query and key features.
        # Self attention scores are scaled down by hidden dimension square
        # root to avoid overflow.
        # `q` Input  shape: `(Head, 1, K)`.
        # `k` Input  shape: `(B, Head, S, K)`.
        # Output shape: `(B, Head, 1, S)`.
        # print(q.shape)
        # print(k.shape)
        attn = q @ k.transpose(-1, -2) / math.sqrt(x.size(-1))

        # Mask parts of attention scores by replacing with large negative
        # values.
        # Input  shape: `(B, Head, 1, S)`.
        # Output shape: `(B, Head, 1, S)`.
        batch_tk_mask = batch_tk_mask.repeat(self.n_head, 1, 1, 1)

        batch_tk_mask = batch_tk_mask.transpose(0, 1)
        batch_tk_mask = batch_tk_mask.transpose(-1, -2)
        # print(attn.shape)
        # print(batch_tk_mask.shape)
        attn.masked_fill_(batch_tk_mask, -1e9)

        # Softmax normalize on attention scores.
        # Large negative values will be closed to zero after normalization.
        # Input  shape: `(B, Head, 1, S)`.
        # Output shape: `(B, Head, 1, S)`.
        attn = F.softmax(attn, dim=-1)

        # Use attention scores to calculate weighted sum on value features.
        # Then perform one more linear tranformation on weighted sum.
        # Finally dropout transformed features.
        # `attn` Input  shape: `(B, Head, 1, S)`.
        # `v` Input  shape: `(B, Head, S, k)`.
        # Output shape: `(B, Head, 1, K)`.
        output = attn @ v

        # Input  shape: `(B, Head, 1, K)`.
        # Output shape: `(B, 1, Hid)`.
        output = output.transpose(1, 2).contiguous()
        output = output.view(B, -1, self.n_head * self.d_k)

        # Output shape: `(B, Hid)`.
        return self.linear(output.squeeze(1))


class Encoder(nn.Module):
    def __init__(self, d_emb: int, p_hid: float):
        super().__init__()
        encoderlayer = nn.TransformerEncoderLayer(d_emb, 4)
        self.linear = nn.Linear(d_emb, d_emb)
        self.pe = PositionalEncoding(d_emb, p_hid)
        self.attn_emb = MultiHeadAttention(d_emb, 4)
        self.layernorm1 = nn.LayerNorm(d_emb)
        self.layernorm2 = nn.LayerNorm(d_emb)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Shape: [B, S, H]
        emb = self.layernorm1(self.linear(x))

        # Shape: [B, S, H]
        emb = self.pe(emb)

        # Shape: [B, H]
        emb = self.layernorm2(self.attn_emb(emb, mask))

        return emb
