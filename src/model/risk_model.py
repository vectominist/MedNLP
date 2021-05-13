import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.encoder import Encoder

class Risk_Classifier(nn.Module):
    def __init__(self, d_emb: int, p_hid: float, n_layers: int):
        super().__init__()
        hid = []
        self.l0 = nn.Linear(d_emb, d_emb)
        for _ in range(n_layers):
            hid.append(nn.Linear(in_features=d_emb, out_features=d_emb))
            hid.append(nn.ReLU())
            hid.append(nn.Dropout(p=p_hid))
        self.hid = nn.Sequential(*hid)
        self.l1 = nn.Linear(d_emb, d_emb//2)
        self.dropout = nn.Dropout(p_hid)
        self.l2 = nn.Linear(d_emb//2, 1)

    def forward(self, document: torch.Tensor) -> torch.Tensor:
        # print("decoding risk....")
        output = document
        output = self.l0(output)

        output = self.hid(output)
        #　Linear layer
        # Input shape: `(B, E)`
        # Ouput shape: `(B, E//2)`
        # output = F.relu(self.l1(document))
        output = F.relu(self.l1(output))

        #　Dropout
        # Input shape: `(B, E//2)`
        # Ouput shape: `(B, E//2)`
        #output = self.dropout(output)

        #　Linear layer
        # Input shape: `(B, E//2)`
        # Ouput shape: `(B, 1)`
        output = torch.sigmoid(self.l2(output))

        return output.squeeze(-1)


class risk_model(nn.Module):
    def __init__(self, d_emb: int, n_layers: int, p_hid: float):
        super().__init__()
        word_embedding = np.load("data/embeddings.npy")
        word_embedding = torch.FloatTensor(word_embedding)
        self.embedding = nn.Embedding.from_pretrained(word_embedding, freeze=True, padding_idx=0)
        self.word_encoder = Encoder(d_emb, p_hid)
        self.encoder = Encoder(d_emb, p_hid)
        self.risk = Risk_Classifier(d_emb, p_hid, n_layers)

    def forward(self, document):
        # Embedding layer
        # Shape: [B, `max_doc_len`, `max_sent_len`, E]
        doc = self.embedding(document)
        w_mask, s_mask = self.create_mask(document)
        temp = []

        # Sentence embedding
        # Shape: [B, `max_doc_len`, E]
        for i in range(doc.shape[0]):
            temp.append(self.word_encoder(doc[i], w_mask[i]))
        doc = torch.stack(temp)

        # Document embedding
        # Input shape: [B, `max_doc_len`, E]
        # Output shape: [B, E]
        doc = self.encoder(doc, s_mask)

        risk_output = self.risk(doc)

        return risk_output

    def create_mask(self, batch_prev_tkids: torch.Tensor) -> torch.Tensor:
        # Create padding self attention masks.
        # Shape: [B, `max_doc_len`, `max_sent_len`, 1]
        # Output dtype: `torch.bool`.
        w_pad_mask = batch_prev_tkids == 0
        w_pad_mask = w_pad_mask.unsqueeze(-1)

        s_pad_mask = batch_prev_tkids.sum(dim=-1)
        s_pad_mask = s_pad_mask == 0
        s_pad_mask = s_pad_mask.unsqueeze(-1)

        return w_pad_mask, s_pad_mask

    def loss_fn(self, document, risk):
        pred_risk = self(document)
        pred_risk = pred_risk.reshape(-1)
        risk = risk.reshape(-1)
        return F.binary_cross_entropy(pred_risk, risk)
