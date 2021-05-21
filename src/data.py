'''
Datasets
'''

from transformers import AutoTokenizer, BertTokenizerFast
import csv
import json
import torch
from torch.utils.data.dataset import Dataset
import unicodedata
from dataset import split_sent
import tqdm
import numpy as np
from augmentation import (
    sentence_random_removal,
    sentence_random_swap,
    EDA)
from util.text_normalization import normalize_sent_with_jieba
from util.lm_normalizer import merge_chinese
from opencc import OpenCC

cc = OpenCC('s2t')  # simplified to traditional

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
tokenizer_risk = BertTokenizerFast.from_pretrained('bert-base-chinese')

choice2int = {
    'A': 0, 'B': 1, 'C': 2,
    'Ａ': 0, 'Ｂ': 1, 'Ｃ': 2
}


def tokenize(x, max_length):
    return tokenizer(
        x, return_tensors="pt", padding="max_length",
        truncation="longest_first", max_length=max_length).input_ids


class ClassificationDataset(Dataset):
    '''
        Dataset for classification
    '''

    def __init__(self, path, split='train', val_r=10,
                 rand_remove=False, rand_swap=False, eda=False):
        assert split in ['train', 'val', 'dev', 'test']

        self.path = path
        self.split = split
        max_doc_len = 120

        with open(path, 'r') as fp:
            data = []
            rows = csv.reader(fp)
            for i, row in enumerate(rows):
                if i == 0:
                    continue
                idx, sent = int(row[1]), row[2]
                sent = normalize_sent_with_jieba(sent)
                sent = sent[:max_doc_len] + [""] * \
                    max(0, max_doc_len - len(sent))
                sent = [merge_chinese(' '.join(s)) for s in sent]
                if split in ['train', 'val']:
                    label = int(row[3])
                    data.append((idx, sent, label))
                else:
                    data.append((idx, sent))

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
                truncation="longest_first", max_length=50)
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
                truncation="longest_first", max_length=50)
            return item


class QADataset(Dataset):
    '''
        Dataset for QA
    '''

    def __init__(self, path, split='train', val_r=10,
                 max_q_len=20, max_c_len=20, max_doc_len=170, max_sent_len=70):
        assert split in ['train', 'val', 'dev', 'test']

        self.path = path
        self.split = split

        with open(path, 'r') as fp:
            data_list = json.load(fp)
            assert type(data_list) == list

            data = []
            print("Reading qa data ...")
            data_list_bar = tqdm.tqdm(data_list, ncols=70)
            for i, d in enumerate(data_list_bar):
                idx = d['id']
                article_idx = d['article_id']
                text = d['text']
                stem = d['question']['stem']
                choices = [c['text'] for c in d['question']['choices']]
                d['answer'] = d['answer'].strip()
                if split in ['train', 'val']:
                    if d['answer'] not in choice2int.keys():
                        # print(d['answer'])
                        answer = [k for k in range(
                            len(choices)) if choices[k] == d['answer']][0]
                    else:
                        answer = choice2int[d['answer']]

                text = unicodedata.normalize("NFKC", text)
                text = ["".join(i) for i in split_sent(text)]
                text = text[:max_doc_len] + [""] * \
                    max(0, max_doc_len - len(text))
                stem = unicodedata.normalize("NFKC", stem)
                choices = [unicodedata.normalize("NFKC", i) for i in choices]

                text = tokenize(text, max_sent_len)
                stem = tokenize(stem, max_q_len).squeeze()
                choices = tokenize(choices, max_c_len)

                if split in ['train', 'val']:
                    one_hot_answer = np.zeros((3,), dtype=np.float32)
                    one_hot_answer[answer] = 1
                    data.append((idx, text, stem, choices, one_hot_answer))
                else:
                    data.append((idx, text, stem, choices))
            data_list_bar.close()
        if split == 'train':
            data = [data[i] for i in range(len(data)) if i % val_r != 0]
        elif split == 'val':
            data = [data[i] for i in range(len(data)) if i % val_r == 0]

        self.data = data

        print('Found {} samples for {} set of the classifcation task'
              .format(len(self.data), split))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # train & val : idx, paragraph, stem, choices, answer
        # dev & test : idx, paragraph, stem, choices
        return self.data[index]


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
            truncation="longest_first", max_length=50, return_special_tokens_mask=True)
        return {key: val[0] for key, val in tokens.items()}


class MultiTaskDataset(Dataset):
    '''
        Dataset for multitask learning (risk pred + qa)
        This dataset is only designed for training.
    '''

    def __init__(self, path_risk, path_qa, split='train', val_r=10, val_mode='risk',
                 rand_remove=False, rand_swap=False, eda=False):
        assert split in ['train', 'val']
        assert val_mode in ['risk', 'qa']

        self.path_risk = path_risk
        self.path_qa = path_qa
        self.val_mode = val_mode
        self.split = split
        max_doc_len = 120

        # Read Risk Prediction data
        # len(data) == number of articles
        with open(path_risk, 'r') as fp:
            risk_data = []
            rows = csv.reader(fp)
            for i, row in enumerate(rows):
                if i == 0:
                    continue
                sent = row[2]
                sent = normalize_sent_with_jieba(sent)
                sent = sent[:max_doc_len] + [""] * \
                    max(0, max_doc_len - len(sent))
                sent = [merge_chinese(' '.join(s)) for s in sent]
                risk_data.append({
                    'id': int(row[1]),
                    'doc': sent,
                    'label': int(row[3])
                })

        def normalize(sent: str) -> str:
            # Helper function for normalization
            sent = normalize_sent_with_jieba(sent, split=False, reduce=False)
            return merge_chinese(' '.join(sent[0]))

        # Read QA data
        with open(path_qa, 'r') as fp:
            data_list = json.load(fp)
            qa_data = [[] for i in range(len(risk_data))]
            for i, d in enumerate(data_list):
                idx = d['article_id']
                stem = normalize(d['question']['stem'])
                choices = [normalize(c['text'])
                           for c in d['question']['choices']]
                d['answer'] = d['answer'].strip()
                if d['answer'] not in choice2int.keys():
                    answer = [k for k in range(3)
                              if d['question']['choices'][k]['text'] == d['answer']][0]
                else:
                    answer = choice2int[d['answer']]

                qa_data[idx - 1].append(
                    {
                        'stem': stem,
                        'choices': choices,
                        'answer': answer
                    }
                )

        assert len(risk_data) == len(qa_data), (len(risk_data), len(qa_data))
        if split == 'train':
            risk_data = [risk_data[i]
                         for i in range(len(risk_data))
                         if (i + 1) % val_r != 0]
            qa_data = [qa_data[i]
                       for i in range(len(qa_data))
                       if (i + 1) % val_r != 0]
        elif split == 'val':
            risk_data = [risk_data[i]
                         for i in range(len(risk_data))
                         if (i + 1) % val_r == 0]
            qa_data = [qa_data[i]
                       for i in range(len(qa_data))
                       if (i + 1) % val_r == 0]

        self.risk_data = risk_data
        self.qa_data = qa_data

        print('Found {} samples for {} set of risk pred'
              .format(len(self.risk_data), split))
        print('Found {} samples for {} set of QA'
              .format(sum([len(q) for q in self.qa_data]), split))

        self.rand_remove = rand_remove
        self.rand_swap = rand_swap
        self.eda = eda
        if rand_remove:
            print('Performing random sentence removal for data augmentation')
        if rand_swap:
            print('Performing random sentence swap for data augmentation')
        if eda:
            print('Performing easy data augmentation')

    def get_ids(self):
        return [d['id'] for d in self.risk_data]

    def __len__(self):
        return len(self.risk_data)

    def __getitem__(self, index):
        # 1. get doc
        sents = self.risk_data[index]['doc']
        if self.eda:
            sents = [EDA(s) for s in sents]
        item = tokenizer_risk(
            sents, return_tensors="pt", padding="max_length",
            truncation="longest_first", max_length=50)
        if self.rand_swap:
            item = sentence_random_swap(item)
        if self.rand_remove:
            item = sentence_random_removal(item)

        # 2. get labels for risk pred
        item['labels_risk'] = torch.tensor(self.risk_data[index]['label'])

        # 3. get stem and choices for qa
        qa_idx = np.random.randint(0, len(self.qa_data[index]))
        stem = tokenizer_risk(
            self.qa_data[index][qa_idx]['stem'], return_tensors="pt",
            padding="max_length", truncation="longest_first", max_length=50)
        item['stem'] = stem['input_ids'].squeeze(0)
        item['attention_mask_stem'] = stem['attention_mask'].squeeze(0)
        choices = [
            tokenizer_risk(c, return_tensors="pt", padding="max_length",
                           truncation="longest_first", max_length=50)
            for c in self.qa_data[index][qa_idx]['choices']
        ]
        item['choice'] = torch.cat(
            [c['input_ids'] for c in choices], dim=0)
        item['attention_mask_choice'] = torch.cat(
            [c['attention_mask'] for c in choices], dim=0)

        # 4. get labels for qa
        item['labels_qa'] = torch.tensor(self.qa_data[index][qa_idx]['answer'])

        if self.split == 'val':
            if self.val_mode == 'risk':
                item['labels'] = item['labels_risk']
            else:
                item['labels'] = item['labels_qa']
            del item['labels_risk']
            del item['labels_qa']

        return item


class QADataset2(Dataset):
    '''
        Dataset for QA (new)
    '''

    def __init__(self, path, split='train', val_r=10,
                 rand_remove=False, rand_swap=False, eda=False):
        assert split in ['train', 'val', 'dev', 'test']

        self.path = path
        self.split = split
        max_doc_len = 120

        def normalize(sent: str) -> str:
            # Helper function for normalization
            sent = normalize_sent_with_jieba(sent, split=False, reduce=False)
            return merge_chinese(' '.join(sent[0]))

        # Read QA data
        with open(path, 'r') as fp:
            data_list = json.load(fp)
            data = []
            for i, d in enumerate(data_list):
                idx = d['id']
                sent = normalize_sent_with_jieba(d['text'])
                sent = sent[:max_doc_len] + [""] * \
                    max(0, max_doc_len - len(sent))
                sent = [merge_chinese(' '.join(s)) for s in sent]
                stem = normalize(d['question']['stem'])
                choices = [normalize(c['text'])
                           for c in d['question']['choices']]
                if split in ['train', 'val']:
                    d['answer'] = d['answer'].strip()
                    if d['answer'] not in choice2int.keys():
                        answer = [k for k in range(3)
                                if d['question']['choices'][k]['text'] == d['answer']][0]
                    else:
                        answer = choice2int[d['answer']]
                else:
                    answer = None
                data.append(
                    {
                        'id': idx,
                        'article_id': d['article_id'],
                        'doc': sent,
                        'stem': stem,
                        'choices': choices,
                        'answer': answer
                    }
                )

        if split == 'train':
            data = [data[i]
                       for i in range(len(data))
                       if data[i]['article_id'] % val_r != 0]
        elif split == 'val':
            data = [data[i]
                       for i in range(len(data))
                       if data[i]['article_id'] % val_r == 0]

        self.data = data

        print('Found {} samples for {} set of QA'
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

    def get_ids(self):
        return [d['id'] for d in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 1. get doc
        sents = self.data[index]['doc']
        if self.eda:
            sents = [EDA(s) for s in sents]
        item = tokenizer_risk(
            sents, return_tensors="pt", padding="max_length",
            truncation="longest_first", max_length=50)
        if self.rand_swap:
            item = sentence_random_swap(item)
        if self.rand_remove:
            item = sentence_random_removal(item)

        # 2. get stem and choices for qa
        stem = EDA(self.data[index]['stem'])
        stem = tokenizer_risk(
            stem, return_tensors="pt",
            padding="max_length", truncation="longest_first", max_length=50)
        item['stem'] = stem['input_ids'].squeeze(0)
        item['attention_mask_stem'] = stem['attention_mask'].squeeze(0)
        choices = [
            tokenizer_risk(c, return_tensors="pt", padding="max_length",
                           truncation="longest_first", max_length=50)
            for c in self.data[index]['choices']
        ]
        item['choice'] = torch.cat(
            [c['input_ids'] for c in choices], dim=0)
        item['attention_mask_choice'] = torch.cat(
            [c['attention_mask'] for c in choices], dim=0)

        # 3. get labels
        if self.split in ['train', 'val']:
            item['labels'] = torch.tensor(self.data[index]['answer'])

        return item


if __name__ == '__main__':
    '''
        Debug code
    '''
    # cl_dataset = ClassificationDataset(
    # 'data/Train_risk_classification_ans.csv', 'train')
    # print(cl_dataset[0])

    # qa_dataset = QADataset(
    #     'data/Train_qa_ans.json', 'train')
    # print(qa_dataset[0])

    mtl_dataset = MultiTaskDataset(
        '../data/Train_risk_classification_ans.csv',
        '../data/Train_qa_ans.json',
        'train'
    )
    print(mtl_dataset[0])
