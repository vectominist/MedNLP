'''
    Risk evaluation w/ SentenceBERT
'''
from transformers import AutoModel
import torch
from torch import nn
from model.encoder import Encoder


def mean_pooling(model_output, attention_mask):
    '''
        Mean Pooling - Take attention mask into account for correct averaging
        Ref: https://www.sbert.net/examples/applications/computing-embeddings/README.html
    '''
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class SBertRiskPredictor(nn.Module):
    '''
        Sentence Bert (or other pre-trained Bert) for Risk Prediction
    '''

    def __init__(
            self,
            model_name, post_encoder_type='transformer', d_model=312,
            sent_aggregate_type='mean_pool'):
        super(SBertRiskPredictor, self).__init__()
        self.sentsence_encoder = AutoModel.from_pretrained(model_name)

        self.sent_aggregate_type = sent_aggregate_type
        assert sent_aggregate_type in \
            ['mean_pool', 'cls', 'attention']
        if self.sent_aggregate_type == 'attention':
            self.agg_attention = Encoder(d_model, 0.1, 8)

        assert post_encoder_type in \
            ['transformer', 'gru', 'lstm', 'none'], post_encoder_type
        if post_encoder_type == 'transformer':
            self.post_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model, 8, 1024, dropout=0.1), 2)
        elif post_encoder_type == 'gru':
            self.post_encoder = nn.GRU(
                d_model, d_model // 2, 2, dropout=0.1,
                batch_first=True, bidirectional=True)
        elif post_encoder_type == 'lstm':
            self.post_encoder = nn.LSTM(
                d_model, d_model // 2, 2, dropout=0.1,
                batch_first=True, bidirectional=True)
        else:
            self.post_encoder = None

        self.attention = Encoder(d_model, 0.1, 8)
        self.pred_head = nn.Linear(d_model, 2)
        self.label_smoothing = 0.1

    def forward(self, **inputs):
        # input size = B x Sentences x Length

        w_mask, s_mask = self.create_mask(inputs['attention_mask'])
        # w_mask: B x S x L x 1
        # s_mask: B x S x 1

        sent_lens = (inputs['attention_mask'].sum(2) > 2).long().sum(1)
        B, S, L = inputs['input_ids'].shape
        for key, val in inputs.items():
            inputs[key] = val.reshape(B * S, L)

        out = self.sentsence_encoder(**inputs)  # B*S x L x D
        if self.sent_aggregate_type == 'mean_pool':
            sent_embs = mean_pooling(out, inputs['attention_mask'])
        elif self.sent_aggregate_type == 'cls':
            sent_embs = out[0][:, 0, :]
        elif self.sent_aggregate_type == 'attention':
            sent_embs = self.agg_attention(out[0], w_mask.reshape(B * S, L, 1))
        sent_embs = sent_embs.reshape(B, S, -1)

        if type(self.post_encoder) in [nn.GRU, nn.LSTM]:
            sent_embs, _ = self.post_encoder(sent_embs)
        elif type(self.post_encoder) == nn.TransformerEncoder:
            sent_embs = self.post_encoder(
                self.attention.pe(sent_embs).transpose(0, 1),
                src_key_padding_mask=s_mask.squeeze(2)).transpose(0, 1)

        h_repr = self.attention(sent_embs, s_mask)
        prediction = self.pred_head(h_repr)

        outputs = {'logits': prediction}

        if inputs.get('labels', None):
            if self.label_smoothing == 0.:
                loss = nn.CrossEntropyLoss()(prediction, inputs['labels'])
            else:
                target = torch.full_like(
                    prediction, self.label_smoothing)
                target.scatter_(1, inputs['labels'], 1. - self.label_smoothing)
                pred = torch.log_softmax(prediction, dim=1)
                loss = (pred * target).sum() / B

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
        s_pad_mask = s_pad_mask <= 2
        s_pad_mask = s_pad_mask.unsqueeze(-1)

        return w_pad_mask, s_pad_mask
