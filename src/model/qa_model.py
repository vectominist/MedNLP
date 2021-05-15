import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.encoder import Encoder
from transformers import AutoModel 

class QA_Classifier(nn.Module):
    def __init__(self, d_emb: int, p_hid: float, n_layers: int):
        super().__init__()
        self.l1 = nn.Linear(3*d_emb, d_emb)
        self.dropout = nn.Dropout(p_hid)

        hid = []
        for _ in range(n_layers):
            hid.append(nn.Linear(in_features=d_emb, out_features=d_emb))
            hid.append(nn.ReLU())
            hid.append(nn.Dropout(p=p_hid))
        self.hid = nn.Sequential(*hid)
        self.l2 = nn.Linear(d_emb, 1)

    def forward(
        self,
        document: torch.Tensor,
        question: torch.Tensor,
        choice: torch.Tensor
    ) -> torch.Tensor:
        # print("decoding QA......")
        # Concatenates `document embedding`, `question embedding`
        # and `choice embeding`
        # Input shape: `(B, E)`, `(B, E)`, `(B, E)`
        # Ouput shape: `(B, 3*E)`
        # print(f"doc:{document.shape}, q:{question.shape}, c:{choice.shape}")
        output = torch.cat((document, question, choice), -1)

        #　Linear layer
        # Input shape: `(B, 3*E)`
        # Ouput shape: `(B, E)`
        output = F.relu(self.l1(output))

        #　Dropout
        # Input shape: `(B, E)`
        # Ouput shape: `(B, E)`
        output = self.dropout(output)

        # Hidden layer
        output = self.hid(output)

        #　Linear layer
        # Input shape: `(B, E)`
        # Ouput shape: `(B, 1)`
        output = torch.sigmoid(self.l2(output))

        return output


class qa_model(nn.Module):
    def __init__(self, d_emb: int, n_layers: int, p_hid: float, readEmbed=True):
        super().__init__()
        if readEmbed:
            word_embedding = np.load("data/embeddings.npy")
            word_embedding = torch.FloatTensor(word_embedding)
            self.embedding = nn.Embedding.from_pretrained(
                word_embedding, freeze=True, padding_idx=0)
        self.word_encoder = Encoder(d_emb, p_hid)
        self.encoder = Encoder(d_emb, p_hid)
        self.qa = QA_Classifier(d_emb, p_hid, n_layers)
    def embed(self, document, question, choice):
        # Embedding layer
        # Shape: [B, `max_doc_len`, `max_sent_len`, E]
        doc = self.embedding(document)
        # Shape: [B, `max_q_len`, E]
        qst = self.embedding(question)
        # Shape: [B, 3, `max_c_len`, E]
        chs = self.embedding(choice)
        
        # Sentence embedding
        # Shape: [B, `max_doc_len`, E]
        w_mask, _ = self.create_mask(document)
        temp = []
        for i in range(doc.shape[0]):
            temp.append(self.word_encoder(doc[i], w_mask[i]))
        doc = torch.stack(temp)

        # Shape: [B, E]
        w_mask, _ = self.create_mask(question)
        qst = self.word_encoder(qst, w_mask)

        # Shape: [3, B, E]
        chs = chs.transpose(0, 1)
        w_mask, _ = self.create_mask(choice.transpose(0, 1))
        chs_temp = [self.word_encoder(chs[i], w_mask[i]) for i in range(chs.shape[0])]
        chs = torch.stack(chs_temp)

        return doc, qst, chs

    def forward(self, document, question, choice):
        doc, qst, chs = self.embed(document, question, choice)

        # Document embedding
        # Input shape: [B, `max_doc_len`, E]
        # Output shape: [B, E]
        _, s_mask = self.create_mask(document)
        doc = self.encoder(doc, s_mask)

        qa_output = []
        for i in range(chs.shape[0]):
            qa_output.append(self.qa(doc, qst, chs[i]))
        qa_output = torch.cat(qa_output, dim=-1)

        return qa_output

    def create_mask(self, batch_prev_tkids: torch.Tensor) -> torch.Tensor:
        # Create padding self attention masks.
        # Shape: [B, `max_doc_len`, `max_sent_len`, 1]
        # Output dtype: `torch.bool`.
        w_pad_mask = batch_prev_tkids != 0
        w_pad_mask = w_pad_mask.unsqueeze(-1)

        s_pad_mask = batch_prev_tkids.sum(dim=-1)
        s_pad_mask = s_pad_mask != 0
        s_pad_mask = s_pad_mask.unsqueeze(-1)

        return w_pad_mask, s_pad_mask

    def loss_fn(self, document, question, choice, qa):
        pred_qa = self(document, question, choice)
        pred_qa = pred_qa.reshape(-1)
        qa = qa.reshape(-1)
        return F.binary_cross_entropy(pred_qa, qa)

class qa_model_hugginggface(qa_model):
    def __init__(self, *args, **kargs):
        super().__init__(768, *args, **kargs, readEmbed=False)
        self.embed_model = AutoModel.from_pretrained("bert-base-chinese")
        self.DisableEncoderTrain()
    def embed(self, document, question, choice):
        batch_size, doc_len, sent_len = document.shape
        
        doc = torch.stack([self.embed_model(i).last_hidden_state for i in document])
        doc = doc[:,:,0,:] # (B,`max_doc_len`,E)

        qst = self.embed_model(question).last_hidden_state[:,0,:] # (B, E)

        chs = torch.stack([self.embed_model(i).last_hidden_state for i in choice])
        chs = chs[:,:,0,:] #(B,3,E)
        chs = chs.transpose(0,1) #(3,B,E)
        
        return doc, qst, chs
    def EnableEncoderTrain(self, enable = True):
        for name, param in self.embed_model.named_parameters():
            if 'classifier' not in name: # classifier layer
                param.requires_grad = enable
    def DisableEncoderTrain(self):
        self.EnableEncoderTrain(False)
