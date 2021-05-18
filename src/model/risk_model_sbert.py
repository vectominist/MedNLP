'''
    Risk evaluation w/ SentenceBERT
'''
from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
import math

# Mean Pooling - Take attention mask into account for correct averaging
# Ref: https://www.sbert.net/examples/applications/computing-embeddings/README.html


def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class DocumentAttention(nn.Module):
    def __init__(self, dim, att_dim):
        super(DocumentAttention, self).__init__()
        self.rnn = nn.GRU(dim, att_dim, 1, batch_first=True)
        self.w_attention = nn.Linear(att_dim, att_dim)
        self.w_post_layer = nn.Linear(2 * att_dim, dim)

    def forward(self, x, x_len):
        # x: B x S x D
        # x_len: B
        B = len(x_len)
        h, _ = self.rnn(x)  # h: B x S x D'
        align = (h[range(B), x_len, :].unsqueeze(1) * self.w_attention(h)).sum(2)
        # align: B x S
        for b in range(B):
            align[b, x_len[b]:] = -math.inf
        align = torch.softmax(align, dim=1)
        context = (h * align.unsqueeze(2)).sum(1)  # context: B x D'
        h_repr = torch.cat([h[range(B), x_len, :], context], dim=1)
        h_repr = self.w_post_layer(h_repr)
        # context-aware utterance representation h_repr: B x D

        return h_repr


class SBertRiskPredictor(nn.Module):
    def __init__(self, model_name, att_dim=256):
        super(SBertRiskPredictor, self).__init__()
        self.sentsence_encoder = AutoModel.from_pretrained(model_name)
        self.attention = DocumentAttention(768, att_dim)
        self.pred_head = nn.Sequential(
            nn.LayerNorm(768, eps=1e-12),
            nn.Linear(768, 2)
        )

    def forward(self, **inputs):
        # print(inputs['input_ids'].shape)
        # input size = B x Sentences x Length
        sent_lens = (inputs['attention_mask'].sum(2) > 2).long().sum(1)
        B, S, L = inputs['input_ids'].shape
        for key, val in inputs.items():
            inputs[key] = val.reshape(B * S, L)

        out = self.sentsence_encoder(**inputs)
        # print(out[0].shape)
        sent_embs = mean_pooling(out, inputs['attention_mask'])
        sent_embs = sent_embs.reshape(B, S, -1)

        h_repr = self.attention(sent_embs, sent_lens - 1)
        prediction = self.pred_head(h_repr)

        outputs = {
            'logits': prediction
        }

        if inputs.get('labels', None):
            loss = nn.CrossEntropyLoss()(prediction, inputs['labels'])
            outputs['loss'] = loss
            outputs['labels'] = inputs['labels']

        return outputs