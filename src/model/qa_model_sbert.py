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
        
        self.attention = Encoder(312, 0.1)
        self.pred_head = nn.Sequential(
            nn.Linear(312, 312),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(312, 156),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(156, 1)
        )

    def forward(self, **inputs):
        # input size = B x 3 x Length
        
        B, _, L = inputs['input_ids'].shape
        for key, val in inputs.items():
            inputs[key] = val.reshape(B * 3, L)
        
        out = self.encoder(**inputs).reshape(B, 3, 312)
        prediction = self.pred_head(out).squeeze(2)  # B x 3

        outputs = {'logits': prediction}
        if inputs.get('labels', None):
            loss = nn.CrossEntropyLoss()(prediction, inputs['labels'])
            outputs['loss'] = loss
            outputs['labels'] = inputs['labels']

        return outputs
