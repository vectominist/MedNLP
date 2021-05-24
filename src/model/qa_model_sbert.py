'''
    QA w/ SentenceBERT
'''
from transformers import AutoModel
import torch
from torch import nn


class SBertQA(nn.Module):
    '''
        Sentence Bert (or other pre-trained Bert) for Multichoice QA
        This is a VERY SIMPLE implementation.
    '''

    def __init__(self, model_name):
        super(SBertQA, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
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
            inputs[key] = val.reshape(B * C, L)

        out = self.encoder(**inputs)
        out = out[0].reshape(B, C, L, 312)[:, :, 0, :]
        prediction = self.pred_head(out).squeeze(2)  # B x C

        if C != 3:
            prediction = prediction.reshape(B, C // 3, 3)  # B x n_chunks x 3
            if self.training:
                # Training
                loss = 0
                outputs = {'logits': prediction[:, 0, :]}
                for b in range(B):
                    loss += nn.CrossEntropyLoss()(
                        prediction[b, :inputs['n_chunks'][b], :],
                        inputs['labels'][b].repeat(inputs['n_chunks'][b]))
                loss = loss / B
                outputs['loss'] = loss
                outputs['labels'] = inputs['labels']
            else:
                # Testing or Validation
                final_logit_idx = []
                for b in range(B):
                    # only choose the largest logits
                    max_logit_idx = (prediction[b, :inputs['n_chunks'][b], :]
                                     .max(1).argmax())
                    final_logit_idx.append(max_logit_idx.cpu().item())
                prediction = prediction[range(B), final_logit_idx, :]
                outputs = {'logits': prediction}
                if inputs.get('labels', None):
                    # Validation
                    loss = nn.CrossEntropyLoss()(prediction, inputs['labels'])
                    outputs['loss'] = loss
                    outputs['labels'] = inputs['labels']
        else:
            outputs = {'logits': prediction}
            if inputs.get('labels', None):
                loss = nn.CrossEntropyLoss()(prediction, inputs['labels'])
                outputs['loss'] = loss
                outputs['labels'] = inputs['labels']

        return outputs
