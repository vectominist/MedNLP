'''
    QA w/ SentenceBERT
'''
from transformers import AutoModel
import torch
from torch import nn
from model.encoder import Encoder
from torch_multi_head_attention import MultiHeadAttention

class SBertQA(nn.Module):
    '''
        Sentence Bert (or other pre-trained Bert) for Multichoice QA
        This is a VERY SIMPLE implementation.
    '''

    def __init__(self, model_name, hidden_dim):
        super(SBertQA, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        # self.attention = Encoder(hidden_dim, 0.1)
        self.attention = MultiHeadAttention(hidden_dim, 1)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # self.post_encoder = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(312, 8, 1024, dropout=0.1), 2)
        # self.pred_head = nn.Sequential(
        #     nn.Dropout(0.1),
        #     nn.Linear(hidden_dim, 312),
        #     nn.Tanh(),
        #     nn.Dropout(0.1),
        #     nn.Linear(312, 1)
        # )
        for name, param in self.encoder.named_parameters():
            if 'classifier' not in name: # classifier layer
                param.requires_grad = False

    def forward(self, **inputs):
        '''
            input size:
                B x 3 x L -> truncated document
                B x 3C x L -> split into sub-documents
        '''
        # input size = B x 3 x Length
        shape = dict()
        for k in ['seq', 'chs', 'stem']:
            B, C, L = inputs[k].shape
            shape[k] = (B, C, L)
        # for key, val in inputs.items():
        #     if val.dim() == 3 and val.shape[0] == B and \
        #             val.shape[1] == C:
        #         inputs[key] = val.reshape(B * C, L)
        for k in ['seq', 'chs', 'stem']:
            B1, C1, _ = shape[k]
            inputs[k] = inputs[k].reshape(B1 * C1, -1)
        # n_chunks = inputs.pop('n_chunks')
        # mask = self.create_mask(n_chunks, shape['seq'][1] // 3)

        labels = inputs.pop('labels') if ('labels' in inputs) else None

        for k in ['seq', 'chs', 'stem']:
            B1, C1, L1 = shape[k]
            out = self.encoder(inputs[k])
            out = out[0].reshape(B1, C1, L1, -1)  # B x C x 312
            inputs[k] = out

        seq = (inputs['seq'][:,:,0,:]
               .reshape(shape['seq'][0], shape['seq'][1] // 3, 3, -1)
               .transpose(1, 2)
               .reshape(shape['seq'][0] * 3, shape['seq'][1] // 3,-1)
               )
        chs = inputs['chs'][:,:,0,:].reshape(shape['chs'][0] * 3, -1)
        stem = inputs['stem'][:,:,0,:].reshape(shape['stem'][0] * 3, 1, -1)
        
        h_repr = self.attention(stem ,seq, seq).squeeze(1)
        prediction = self.cos(h_repr, chs).reshape(shape['chs'][0], 3)
        # FIXME: adding transformer does not improve much
        # out = self.post_encoder(
        #     self.attention.pe(out).transpose(0, 1),
        #     src_key_padding_mask=mask.squeeze(2)).transpose(0, 1)
        # h_repr = self.attention(seq, mask)  # 3B x D
        # h_repr = h_repr.reshape(B, 3, -1)
        # prediction = self.pred_head(h_repr).squeeze(2)

        outputs = {'logits': prediction}
        if labels is not None:
            loss = nn.CrossEntropyLoss()(prediction, labels)
            outputs['loss'] = loss
            outputs['labels'] = labels

        # prediction = self.pred_head(out).squeeze(2)  # B x C

        # if C != 3:
        #     prediction = prediction.reshape(B, C // 3, 3)  # B x n_chunks x 3
        #     if self.training:
        #         # Training
        #         loss = 0
        #         outputs = {'logits': prediction[:, 0, :]}
        #         for b in range(B):
        #             loss += nn.CrossEntropyLoss()(
        #                 prediction[b, :n_chunks[b], :],
        #                 labels[b].repeat(n_chunks[b]))
        #         loss = loss / B
        #         outputs['loss'] = loss
        #         outputs['labels'] = labels
        #     else:
        #         # Testing or Validation
        #         final_logit_idx = []
        #         for b in range(B):
        #             # only choose the largest logits
        #             max_logit_idx = (prediction[b, :n_chunks[b], :]
        #                              .max(1)[0].argmax())
        #             final_logit_idx.append(max_logit_idx.cpu().item())
        #         prediction = prediction[range(B), final_logit_idx, :]
        #         outputs = {'logits': prediction}
        #         if labels is not None:
        #             # Validation
        #             loss = nn.CrossEntropyLoss()(prediction, labels)
        #             outputs['loss'] = loss
        #             outputs['labels'] = labels
        # else:
        #     outputs = {'logits': prediction}
        #     if labels is not None:
        #         loss = nn.CrossEntropyLoss()(prediction, labels)
        #         outputs['loss'] = loss
        #         outputs['labels'] = labels

        return outputs

    def create_mask(
            self,
            n_chunks: torch.Tensor,
            max_chunks: int = 8) -> torch.Tensor:
        # Create padding self attention masks.
        # Shape: [B, `max_doc_len`, `max_sent_len`, 1]
        # Output dtype: `torch.bool`.

        B = len(n_chunks)
        mask = torch.zeros(
            (B, 3, max_chunks),
            dtype=torch.bool, device=n_chunks.device)

        for b in range(B):
            mask[b, :, n_chunks[b]:] = True

        return mask.reshape(B * 3, -1).unsqueeze(2)  # 3B x C x 1
