'''
    Risk evaluation w/ SentenceBERT
'''

import torch
from torch import nn
import math
from model.risk_model_sbert import DocumentAttention

class Doc2VecRiskPredictor(nn.Module):
    def __init__(self, att_dim=64, doc_dim=256):
        super(Doc2VecRiskPredictor, self).__init__()
        self.attention = DocumentAttention(doc_dim, att_dim)
        self.pred_head = nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm(att_dim, eps=1e-12),
            nn.Linear(att_dim, 2)
        )

    def forward(self, **inputs):
        # input size = B x Sentences x Doc_dim
        sent_embs = inputs['input_ids']
        sent_lens = inputs['input_lens']
        h_repr = self.attention(sent_embs, sent_lens - 1)
        prediction = self.pred_head(h_repr)

        outputs = {'logits': prediction}

        if inputs.get('labels', None):
            loss = nn.CrossEntropyLoss()(prediction, inputs['labels'])
            outputs['loss'] = loss
            outputs['labels'] = inputs['labels']

        return outputs
