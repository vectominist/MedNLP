'''
    QA w/ SentenceBERT
'''
from transformers import AutoModel
import torch
from torch import nn
from model.encoder import Encoder


class SBertQA(nn.Module):
    '''
        Sentence Bert (or other pre-trained Bert) for Multichoice QA
        This is a VERY SIMPLE implementation.
    '''

    def __init__(self, model_name):
        super(SBertQA, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.attention = Encoder(312, 0.1)
        self.post_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(312, 8, 1024, dropout=0.1), 2)
        self.pred_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(312, 312),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(312, 1)
        )

    def forward(self, **inputs):
        '''
            input size:
                B x 3 x L -> truncated document
                B x 3C x L -> split into sub-documents
        '''
        # input size = B x 3 x Length

        B, C, L = inputs['input_ids'].shape
        for key, val in inputs.items():
            if val.dim() == 3 and val.shape[0] == B and \
                    val.shape[1] == C and val.shape[2] == L:
                inputs[key] = val.reshape(B * C, L)
        if C != 3:
            n_chunks = inputs.pop('n_chunks')

        labels = inputs.pop('labels') if ('labels' in inputs) else None

        out = self.encoder(**inputs)
        out = out[0].reshape(B, C, L, 312)[:, :, 0, :]  # B x C x 312

        mask = self.create_mask(n_chunks, C // 3)
        out = (out
               .reshape(B, C // 3, 3, 312)
               .transpose(1, 2)
               .reshape(B * 3, C // 3, 312))
        out = self.post_encoder(
            self.attention.pe(out).transpose(0, 1),
            src_key_padding_mask=mask.squeeze(2)).transpose(0, 1)
        h_repr = self.attention(out, mask)  # 3B x D
        h_repr = h_repr.reshape(B, 3, 312)
        prediction = self.pred_head(h_repr).squeeze(2)

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
