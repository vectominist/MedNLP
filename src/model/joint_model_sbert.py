'''
    Joint Model for Risk Prediction & Multichoice QA
'''
from transformers import AutoModel
import torch
from torch import nn
from model.encoder import Encoder
from model.risk_model_sbert import mean_pooling


class SBertJointPredictor(nn.Module):
    '''
        Bert (or other pre-trained LM) 
        for Risk Prediction & Multichoice QA
    '''

    def __init__(self, model_name, att_dim=128):
        super(SBertJointPredictor, self).__init__()
        self.sentsence_encoder = AutoModel.from_pretrained(model_name)
        self.attention = Encoder(312, 0.1)
        self.risk_pred_head = nn.Sequential(
            nn.Linear(312, 312),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(312, 156),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(156, 2)
        )
        self.qa_pred_head = nn.Sequential(
            nn.Linear(312 * 2, 312),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(312, 156),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(156, 1)
        )
        self.eval_qa_mode = False

    def eval_qa(self, option: bool):
        '''
            Change evaluation task to QA
        '''
        self.eval_qa_mode = option

    def forward(self, **inputs):
        '''
            inputs: {
                'input_ids' : tokenized sentences (B x S x L)
                'attention_mask': attention mask (B x S x L)
                'labels_risk': labels for risk pred (B, optional)
                'stem': question (B x L)
                'attention_mask_stem': attention mask (B x L)
                'choice': answer for QA (B x 3 x L)
                'attention_mask_choice': attention mask (B x 3 x L)
                'labels_qa': labels for qa (B, optional)
            }
        '''
        has_qa = inputs.get('stem') is not None

        # Encode sentences
        _, s_mask = self.create_mask(inputs['attention_mask'])

        B, S, L = inputs['input_ids'].shape
        doc_inputs = {
            'input_ids': inputs['input_ids'].reshape(B * S, L),
            'attention_mask': inputs['attention_mask'].reshape(B * S, L)
        }

        doc_out = self.sentsence_encoder(**doc_inputs)
        doc_embs = mean_pooling(doc_out, doc_inputs['attention_mask'])
        doc_embs = doc_embs.reshape(B, S, -1)  # B x S x D

        # QA forward and calculate prediction
        if has_qa:
            # Either training or testing, this part is necessary for QA
            stem_inputs = {
                'input_ids': inputs['stem'],
                'attention_mask': inputs['attention_mask_stem']
            }
            choice_inputs = {
                'input_ids': inputs['choice'].reshape(B * 3, -1),
                'attention_mask': inputs['attention_mask_choice'].reshape(B * 3, -1)
            }
            stem_out = self.sentsence_encoder(**stem_inputs)
            stem_embs = mean_pooling(stem_out, stem_inputs['attention_mask'])
            choice_out = self.sentsence_encoder(**choice_inputs)
            choice_embs = mean_pooling(
                choice_out, choice_inputs['attention_mask'])
            choice_embs = choice_embs.reshape(B, 3, -1)  # B x 3 x D

            w_mask, _ = self.create_mask(inputs['attention_mask_stem'])
            stem_repr = self.attention(
                doc_embs, w_mask, q=stem_embs, is_qa=True)  # B x D
            cat_repr = torch.cat([
                doc_embs.unsqueeze(1).expand(B, 3, -1),
                choice_embs], dim=2)  # B x 3 x 2D
            qa_prediction = self.qa_pred_head(cat_repr).squeeze(2)
            # qa_prediction: B x 3

        # Risk prediction
        h_repr = self.attention(doc_embs, s_mask)
        prediction = self.risk_pred_head(h_repr)  # B x 2
        outputs = {'logits': prediction}

        if inputs.get('labels', None):
            # Has labels: either training or validation
            if not self.eval_qa_mode:
                loss = nn.CrossEntropyLoss()(prediction, inputs['labels'])
            else:
                # Validation mode for QA only
                assert has_qa
                loss = nn.CrossEntropyLoss()(qa_prediction, inputs['labels'])
                outputs['logits'] = qa_prediction
            outputs['loss'] = loss
            outputs['labels'] = inputs['labels']
        else:
            # Testing mode only
            if has_qa:
                outputs['logits'] = qa_prediction

        if has_qa and inputs.get('labels_qa', None):
            # 'labels_qa' will only occur when training
            qa_loss = nn.CrossEntropyLoss()(qa_prediction, inputs['labels_qa'])
            outputs['loss'] += qa_loss
            # outputs['labels_qa'] = inputs['labels_qa']

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
