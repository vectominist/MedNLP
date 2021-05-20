'''
    Risk evaluation w/ SentenceBERT
'''
from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
import math
from model.encoder import Encoder

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


class DocumentRNN(nn.Module):
    def __init__(self, dim, att_dim):
        super(DocumentRNN, self).__init__()
        self.rnn = nn.GRU(dim, att_dim, 2, batch_first=True, dropout=0.3)

    def forward(self, x, x_len):
        # x: B x S x D
        # x_len: B
        B = len(x_len)
        h, _ = self.rnn(x)  # h: B x S x D'
        
        return h[range(B), x_len]


class DocumentAttention(nn.Module):
    def __init__(self, dim, att_dim, rnn=False):
        super(DocumentAttention, self).__init__()
        if rnn:
            self.rnn = nn.GRU(dim, att_dim, 1, batch_first=True)
        else:
            self.rnn = None
            self.in_layer = nn.Linear(dim, att_dim) if dim != att_dim else None
        self.Wk = nn.Linear(att_dim, att_dim, bias=False)  # key
        self.Wq = nn.Linear(att_dim, att_dim, bias=False)  # query
        self.w_post_layer = nn.Linear(2 * att_dim, att_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, x_len):
        # x: B x S x D
        # x_len: B
        B = len(x_len)
        if self.rnn:
            h, _ = self.rnn(x)  # h: B x S x D'
        else:
            if self.in_layer:
                h = self.in_layer(x)
            else:
                h = x
        h = self.dropout(h)
        align = (self.Wk(h[range(B), x_len, :].unsqueeze(1)) * self.Wq(h)).sum(2)
        # align: B x S
        for b in range(B):
            align[b, x_len[b]:] = -math.inf
        align = torch.softmax(align, dim=1)
        context = (h * align.unsqueeze(2)).sum(1)  # context: B x D'
        h_repr = torch.cat([h[range(B), x_len, :], context], dim=1)
        h_repr = self.dropout(h_repr)
        h_repr = self.w_post_layer(torch.relu(h_repr))
        # context-aware utterance representation h_repr: B x D

        return h_repr


class SBertRiskPredictor(nn.Module):
    '''
        Sentence Bert (or other pre-trained Bert) for Risk Prediction
    '''
    def __init__(self, model_name, att_dim=128):
        super(SBertRiskPredictor, self).__init__()
        self.sentsence_encoder = AutoModel.from_pretrained(model_name)
        # self.attention = DocumentAttention(312, att_dim, rnn=True)
        # self.attention = DocumentRNN(312, att_dim)
        self.attention = Encoder(312, 0.1)
        self.pred_head = nn.Sequential(
            nn.Linear(312, 312),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(312, 156),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(156, 2)
        )

    def forward(self, **inputs):
        # input size = B x Sentences x Length

        _, s_mask = self.create_mask(
            inputs['attention_mask'])  # B x S x L

        sent_lens = (inputs['attention_mask'].sum(2) > 2).long().sum(1)
        B, S, L = inputs['input_ids'].shape
        for key, val in inputs.items():
            inputs[key] = val.reshape(B * S, L)

        out = self.sentsence_encoder(**inputs)
        sent_embs = mean_pooling(out, inputs['attention_mask'])
        sent_embs = sent_embs.reshape(B, S, -1)

        # h_repr = self.attention(sent_embs, sent_lens - 1)
        h_repr = self.attention(sent_embs, s_mask)
        prediction = self.pred_head(h_repr)

        outputs = {'logits': prediction}

        if inputs.get('labels', None):
            loss = nn.CrossEntropyLoss()(prediction, inputs['labels'])
            outputs['loss'] = loss
            outputs['labels'] = inputs['labels']

        return outputs
    
    def create_mask(self, batch_prev_tkids: torch.Tensor) -> torch.Tensor:
        # Create padding self attention masks.
        # Shape: [B, `max_doc_len`, `max_sent_len`, 1]
        # Output dtype: `torch.bool`.
        w_pad_mask = batch_prev_tkids == 0
        w_pad_mask = w_pad_mask.unsqueeze(-1)

        s_pad_mask = batch_prev_tkids.sum(-1)
        s_pad_mask = s_pad_mask == 0
        s_pad_mask = s_pad_mask.unsqueeze(-1)

        return w_pad_mask, s_pad_mask
