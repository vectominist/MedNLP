'''
Datasets
'''

import csv
import json
import torch
from torch.utils.data.dataset import Dataset
from util.text_normalization import normalize_sent_with_jieba
import tqdm
import numpy as np
from augmentation import sentence_random_removal
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


choice2int = {
    'A': 0, 'B': 1, 'C': 2,
    'Ａ': 0, 'Ｂ': 1, 'Ｃ': 2
}


class ClassificationDataset(Dataset):
    '''
        Dataset for classification
    '''

    def __init__(self,
                 path, split='train', val_r=10, rand_remove=False,
                 doc2vec='', max_sentences=170):
        assert split in ['train', 'val', 'dev', 'test']

        self.path = path
        self.split = split
        self.doc2vec_model = Doc2Vec.load(doc2vec)
        self.max_sentences = max_sentences
        print('Using pre-trained Doc2Vec from {}'.format(doc2vec))

        with open(path, 'r') as fp:
            data = []
            rows = csv.reader(fp)
            for i, row in enumerate(rows):
                if i == 0:
                    continue
                idx, sent = int(row[1]), row[2]
                sent = normalize_sent_with_jieba(sent)  # list of lists
                sent = [self.doc2vec_model.infer_vector(s) for s in sent]
                sent = torch.tensor(sent)  # Sentences x Dim
                sent_len = torch.LongTensor(min(sent.shape[1], max_sentences))
                if sent.shape[0] > max_sentences:
                    sent = sent[:max_sentences]
                elif sent.shape[0] < max_sentences:  # pad sequence
                    sent = torch.cat(
                        [sent] + (max_sentences - sent.shape[0]) * [torch.zeros((1, sent.shape[1]))], dim=0)
                if split in ['train', 'val']:
                    label = int(row[3])
                    data.append((idx, sent, sent_len, label))
                else:
                    data.append((idx, sent, sent_len))

        if split == 'train':
            data = [data[i] for i in range(len(data)) if i % val_r != 0]
        elif split == 'val':
            data = [data[i] for i in range(len(data)) if i % val_r == 0]

        self.data = data

        print('Found {} samples for {} set of the classifcation task'
              .format(len(self.data), split))

        self.rand_remove = rand_remove
        if rand_remove:
            print('Performing random sentence removal for data augmentation')

    def get_ids(self):
        return [d[0] for d in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # train & val : idx, paragraph, label
        # dev & test : idx, paragraph
        if self.split in ['train', 'val']:
            item = {
                'input_ids': self.data[index][1],
                'input_lens': self.data[index][2]
            }
            # if self.rand_remove:
            #     item = sentence_random_removal(item)
            item['labels'] = torch.tensor(self.data[index][3])
            return item
        else:
            item = {
                'input_ids': self.data[index][1],
                'input_lens': self.data[index][2]
            }
            # item = {key: val for key, val in self.data[index][1].items()}
            return item
