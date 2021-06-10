'''
    Datasets
'''

import csv
import json
import numpy as np

import tqdm

import torch
from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer, BertTokenizerFast
from augmentation import (
    sentence_random_removal,
    sentence_random_swap,
    EDA)

from util.text_normalization import normalize_sent_with_jieba
from util.lm_normalizer import merge_chinese
import multiprocessing as mp

tokenizer_risk = BertTokenizerFast.from_pretrained('bert-base-chinese')

choice2int = {
    'A': 0, 'B': 1, 'C': 2,
    'Ａ': 0, 'Ｂ': 1, 'Ｃ': 2
}


def crop_doc(sents, max_doc_len=170):
    if len(sents) < max_doc_len:
        return sents + [""] * (max_doc_len - len(sents))
    else:
        return sents[-max_doc_len:]


class ClassificationDataset(Dataset):
    '''
        Dataset for classification
    '''

    def __init__(self, path, split='train', val_r=10,
                 rand_remove=False, rand_swap=False, eda=False):
        assert split in ['train', 'val', 'dev', 'test']

        self.path = path
        self.split = split
        self.max_doc_len = 500
        sent_lens = []

        with open(path, 'r') as fp:
            rows = csv.reader(fp)
            row_list = []
            for i, r in enumerate(rows):
                if i == 0:
                    continue
                row_list.append(r)

        with mp.Pool() as p:
            data = p.starmap(self._preprocess_single_data,
                             enumerate(row_list, start=1))

        # sent_lens = np.array(sent_lens)
        # print('Sentence lengths: avg = {:.1f}, med = {}, min = {}, max = {}'
        #       .format(sent_lens.mean(), np.median(sent_lens), sent_lens.min(), sent_lens.max()))

        if split == 'train':
            data = [data[i] for i in range(len(data)) if (i + 1) % val_r != 0]
        elif split == 'val':
            data = [data[i] for i in range(len(data)) if (i + 1) % val_r == 0]

        self.data = data

        print('Found {} samples for {} set of the classifcation task'
              .format(len(self.data), split))

        self.rand_remove = rand_remove
        self.rand_swap = rand_swap
        self.eda = eda
        if rand_remove:
            print('Performing random sentence removal for data augmentation')
        if rand_swap:
            print('Performing random sentence swap for data augmentation')
        if eda:
            print('Performing easy data augmentation')

    def _preprocess_single_data(self, i, row):
        idx, sent = int(row[1]), row[2]
        sent = normalize_sent_with_jieba(
            sent, reduce=False, max_sent_len=40)
        sent = crop_doc(sent, self.max_doc_len)
        sent = [merge_chinese(' '.join(s)) for s in sent]
        if self.split in ['train', 'val']:
            label = int(row[3])
            return idx, sent, label
        else:
            return idx, sent

    def get_ids(self):
        return [d[0] for d in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # train & val : idx, paragraph, label
        # dev & test : idx, paragraph
        if self.split in ['train', 'val']:
            sents = self.data[index][1]
            if self.eda:
                sents = [EDA(s) for s in sents]
            item = tokenizer_risk(
                sents, return_tensors="pt", padding="max_length",
                truncation="longest_first", max_length=40)
            if self.rand_swap:
                item = sentence_random_swap(item)
            if self.rand_remove:
                item = sentence_random_removal(item)
            item['labels'] = torch.tensor(self.data[index][2])
            return item
        else:
            sents = self.data[index][1]
            item = tokenizer_risk(
                sents, return_tensors="pt", padding="max_length",
                truncation="longest_first", max_length=40)
            return item


class QADatasetRuleBase(Dataset):
    def __init__(self, path):
        self.path = path

        # Read QA data
        with open(path, 'r') as fp:
            data_list = json.load(fp)
        with mp.Pool() as p:
            data = p.starmap(self._preprocess_single_data,
                             enumerate(data_list))

        self.data = data

        print('Found {} samples of QA'.format(len(self.data)))

    def _preprocess_single_data(self, i, d):
        def normalize(sent: str) -> str:
            # Helper function for normalization
            sent = normalize_sent_with_jieba(
                sent, split=False, reduce=False,
                max_sent_len=50, remove_short=False)
            return merge_chinese(' '.join(sent[0]))

        idx = d['id']
        sent = normalize_sent_with_jieba(
            d['text'], reduce=False, max_sent_len=np.inf)
        # sent = crop_doc(sent, max_doc_len)
        sent = [merge_chinese(' '.join(s)) for s in sent]
        stem = normalize(d['question']['stem'])
        choices = [normalize(c['text'])
                   for c in d['question']['choices']]
        if 'answer' in d.keys():
            d['answer'] = d['answer'].strip()
            if d['answer'] not in choice2int.keys():
                answer = [k for k in range(3)
                          if d['question']['choices'][k]['text'] == d['answer']][0]
            else:
                answer = choice2int[d['answer']]
        else:
            answer = None
        return {
            'id': idx,
            'article_id': d['article_id'],
            'doc': sent,
            'stem': stem,
            'choices': choices,
            'answer': answer
        }

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get_ids(self):
        return [d['id'] for d in self.data]


class MLMDataset(Dataset):
    '''
        Dataset for Masked LM
    '''

    def __init__(self, path, split='train', val_r=10, eda=False):
        assert split in ['train', 'val', 'dev', 'test', 'train_all']

        self.path = path
        self.split = split

        with open(path, 'r') as fp:
            data = []
            for line in fp.readlines():
                line = line.strip()  # should be normalized and segmented
                if line == '':
                    continue
                data.append(line)

        if split == 'train':
            data = [data[i] for i in range(len(data)) if i % val_r != 0]
        elif split == 'val':
            data = [data[i] for i in range(len(data)) if i % val_r == 0]

        self.data = data

        print('Found {} samples for {} set of the classifcation task'
              .format(len(self.data), split))

        self.eda = eda
        if eda:
            print('Performing easy data augmentation')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sent = self.data[index]
        if self.eda:
            sent = EDA(sent)
        tokens = tokenizer_risk(
            sent, return_tensors="pt", padding="max_length",
            truncation="longest_first", max_length=40,
            return_special_tokens_mask=True)
        return {key: val[0] for key, val in tokens.items()}


if __name__ == '__main__':
    pass
    '''
        Debug code
    '''
    # cl_dataset = ClassificationDataset(
    # 'data/Train_risk_classification_ans.csv', 'train')
    # print(cl_dataset[0])

    # qa_dataset = QADataset(
    #     'data/Train_qa_ans.json', 'train')
    # print(qa_dataset[0])

    # mtl_dataset = MultiTaskDataset(
    #     '../data/Train_risk_classification_ans.csv',
    #     '../data/Train_qa_ans.json',
    #     'train'
    # )
    # print(mtl_dataset[0])

    # qa_dataset = QADataset3(
    #     '../data/train_qa_tr-dv.json', 'train', doc_splits=8)
    # print(qa_dataset.data[3]['doc_split'])
    # print(qa_dataset.data[3]['doc'])
    # print(qa_dataset[3])
    # doc_lens = [len(' '.join(d['doc'])) for d in qa_dataset.data]
    # doc_lens = np.array(doc_lens)
    # print('Doc lengths : AVG = {:.1f} , MED = {} , MIN = {:.1f} , MAX = {:.1f}'
    #       .format(doc_lens.mean(), np.median(doc_lens), doc_lens.min(), doc_lens.max()))

    # q_lens = [len(d['stem']) for d in qa_dataset.data]
    # q_lens = np.array(q_lens)
    # print('Q lengths : AVG = {:.1f} , MED = {} , MIN = {:.1f} , MAX = {:.1f}'
    #       .format(q_lens.mean(), np.median(q_lens), q_lens.min(), q_lens.max()))

    # c_lens = [max([len(c) for c in d['choices']]) for d in qa_dataset.data]
    # c_lens = np.array(c_lens)
    # print('A lengths : AVG = {:.1f} , MED = {} , MIN = {:.1f} , MAX = {:.1f}'
    #       .format(c_lens.mean(), np.median(c_lens), c_lens.min(), c_lens.max()))

    # Doc lengths : AVG = 1499.7 , MED = 1481.0 , MIN = 351.0 , MAX = 4137.0
    # Q lengths : AVG = 14.5 , MED = 14.0 , MIN = 7.0 , MAX = 29.0
    # A lengths : AVG = 8.5 , MED = 7.0 , MIN = 2.0 , MAX = 36.0
