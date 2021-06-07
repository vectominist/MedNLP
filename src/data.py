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
import numpy as np


tokenizer_qa = BertTokenizerFast.from_pretrained("ckiplab/albert-tiny-chinese")
tokenizer_risk = BertTokenizerFast.from_pretrained('bert-base-chinese')

choice2int = {
    'A': 0, 'B': 1, 'C': 2,
    'Ａ': 0, 'Ｂ': 1, 'Ｃ': 2
}

def is_inv(sent:str):
    for i in ["錯誤","有誤","不正確","不符合","非","不是","不包括","不包含","沒有"]:
        if i in sent:
            return True
    return False


def crop_doc(sents, max_doc_len=170):
    if len(sents) < max_doc_len:
        return sents + [""] * (max_doc_len - len(sents))
    else:
        return sents[-max_doc_len:]

class QADatasetRuleBase(Dataset):
    def __init__(self, path):
        self.path = path

        def normalize(sent: str) -> str:
            # Helper function for normalization
            sent = normalize_sent_with_jieba(
                sent, split=False, reduce=False,
                max_sent_len=50, remove_short=False)
            return merge_chinese(' '.join(sent[0]))

        # Read QA data
        with open(path, 'r') as fp:
            data_list = json.load(fp)
            data = []
            for i, d in enumerate(data_list):
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

        self.data = data

        print('Found {} samples of QA'.format(len(self.data)))
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)
    def get_ids(self):
        return [d['id'] for d in self.data]

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
                sent = normalize_sent_with_jieba(
                    sent, reduce=False, max_sent_len=50)
                sent = crop_doc(sent, max_doc_len)
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
                text = crop_doc(text, max_doc_len)
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
            truncation="longest_first", max_length=512,
            return_special_tokens_mask=True)
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
                sent = crop_doc(sent, max_doc_len)
                sent = [merge_chinese(' '.join(s)) for s in sent]
                risk_data.append({
                    'id': int(row[1]),
                    'doc': sent,
                    'label': int(row[3])
                })

        def normalize(sent: str) -> str:
            # Helper function for normalization
            sent = normalize_sent_with_jieba(
                sent, split=False, reduce=False, max_sent_len=50)
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
        self.tot_qa_data = sum([len(q) for q in self.qa_data])
        self.qa_len_accum = [0] * len(self.qa_data)
        self.index2qa = [0] * self.tot_qa_data
        for i in range(1, len(self.qa_len_accum)):
            self.qa_len_accum[i] = \
                self.qa_len_accum[i - 1] + len(self.qa_data[i - 1])
            for j in range(self.qa_len_accum[i], self.qa_len_accum[i] + len(self.qa_data[i])):
                self.index2qa[j] = i

        print('Found {} samples for {} set of risk pred'
              .format(len(self.risk_data), split))
        print('Found {} samples for {} set of QA'
              .format(self.tot_qa_data, split))

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
        if self.split == 'train' or self.val_mode == 'risk':
            return len(self.risk_data)
        else:
            return sum([len(q) for q in self.qa_data])

    def __getitem__(self, index):
        # 1. get doc
        if self.split == 'train' or self.val_mode == 'risk':
            risk_idx = index
        else:
            risk_idx = self.index2qa[index]
        sents = self.risk_data[risk_idx]['doc']
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
        item['labels_risk'] = torch.tensor(self.risk_data[risk_idx]['label'])

        # 3. get stem and choices for qa
        if self.split == 'train' or self.val_mode == 'risk':
            qa_idx = np.random.randint(0, len(self.qa_data[index]))
        else:
            qa_idx = self.qa_len_accum[self.index2qa[index]] - index
            index = self.index2qa[index]
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
            sent = normalize_sent_with_jieba(
                sent, split=False, reduce=False,
                max_sent_len=50, remove_short=False)
            return merge_chinese(' '.join(sent[0]))

        # Read QA data
        with open(path, 'r') as fp:
            data_list = json.load(fp)
            data = []
            for i, d in enumerate(data_list):
                idx = d['id']
                sent = normalize_sent_with_jieba(
                    d['text'], reduce=False, max_sent_len=70)
                # sent = crop_doc(sent, max_doc_len)
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
        stem = self.data[index]['stem']
        if self.eda:
            stem = EDA(stem)
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


class QADataset3(QADataset2):
    '''
        Dataset for QA (new for single task training)
    '''

    def __init__(self, path, split='train', val_r=10,
                 rand_remove=False, rand_swap=False, eda=False,
                 doc_splits=1):
        super().__init__(path, split, val_r, rand_remove, rand_swap, eda)
        self.doc_splits = doc_splits
        if self.doc_splits > 1:
            max_sub_doc_len = 400
            max_overlap_len = 200
            print('Splitting documents into {} chunks (chunk max chars = {} , chunks overlap chars = {})'
                  .format(doc_splits, max_sub_doc_len, max_overlap_len))
            split_data = []
            for d in self.data:
                sub_docs = []
                stack = []
                curr_len = 0
                for s in d['doc']:
                    if curr_len + len(s) > max_sub_doc_len:
                        sub_docs.append(' '.join(stack))
                        if len(sub_docs) >= doc_splits:
                            break
                        while len(stack) > 0 and curr_len >= max_overlap_len:
                            curr_len -= len(stack[0])
                            stack = stack[1:]
                    stack.append(s)
                    curr_len += len(s)
                if len(sub_docs) < doc_splits:
                    sub_docs.append(' '.join(stack))
                d['doc_n_chunks'] = len(sub_docs)
                if len(sub_docs) < doc_splits:
                    sub_docs += [''] * (doc_splits - len(sub_docs))
                d['doc_split'] = sub_docs

    def __getitem__(self, index):
        # 1. get doc
        if self.doc_splits == 1:
            sents = self.data[index]['doc']
        else:
            sents = self.data[index]['doc_split']
        if self.eda:
            sents = [EDA(s) for s in sents]

        # 2. get stem and choices for qa
        stem = self.data[index]['stem']
        if self.eda:
            stem = EDA(stem)

        # 3. collect input
        # if self.doc_splits == 1:
        #     seq = ['{}[SEP]{}[SEP]{}'.format(' '.join(sents), stem, c)
        #            for c in self.data[index]['choices']]
        # else:
        #     seq = []
        #     for s in sents:
        #         chunk = ['{}[SEP]{}[SEP]{}'.format(' '.join(sents), stem, c)
        #                  for c in self.data[index]['choices']]
        #         seq += chunk
            # seq: list of length 3 * C
        # seq = [s if len(s) <= 518 else s[-518:] for s in seq]

        # 4. tokenize
        # item = tokenizer_risk(seq, return_tensors="pt", padding="max_length",
                              # truncation="longest_first", max_length=512)
        if self.doc_splits == 1:
            seq = [' '.join(sents) for c in self.data[index]['choices']]
        else:
            seq = []
            for s in sents:
                chunk = [' '.join(s) for c in self.data[index]['choices']]
                seq += chunk
        chs = self.data[index]['choices']
        

        seq = tokenizer_qa([stem] * len(seq), seq, add_special_tokens=True, return_tensors="pt", padding="max_length",
                          truncation="longest_first", max_length=512)
        if self.rand_swap:
            seq = sentence_random_swap(seq)
        if self.rand_remove:
            seq = sentence_random_removal(seq)

        chs = tokenizer_qa([stem] * len(chs), chs, add_special_tokens=True, return_tensors="pt", padding="max_length",
                          truncation="longest_first", max_length=30)
        stem = tokenizer_qa([stem] * len(chs), return_tensors="pt", padding="max_length",
                          truncation="longest_first", max_length=30)

        item = dict()
        for key,val in seq.items():
            item["seq_%s" % key] = val
        for key,val in chs.items():
            item["chs_%s" % key] = val
        for key,val in stem.items():
            item["stem_%s" % key] = val
        
        # 5. get labels
        if self.split in ['train', 'val']:
            item['inv'] = torch.tensor([-1])
            if is_inv(stem):
                item['inv'] = torch.tensor([1])
            item['label'] = self.data[index]['answer']

        # 6. add number of chunks of the document
        if self.doc_splits > 1:
            item['n_chunks'] = torch.tensor(self.data[index]['doc_n_chunks'])
        else:
            item['n_chunks'] = torch.tensor([len(self.data[index])])

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

    # mtl_dataset = MultiTaskDataset(
    #     '../data/Train_risk_classification_ans.csv',
    #     '../data/Train_qa_ans.json',
    #     'train'
    # )
    # print(mtl_dataset[0])

    qa_dataset = QADataset3(
        '../data/train_qa_tr-dv.json', 'train', doc_splits=8)
    print(qa_dataset.data[3]['doc_split'])
    print(qa_dataset.data[3]['doc'])
    print(qa_dataset[3])
    doc_lens = [len(' '.join(d['doc'])) for d in qa_dataset.data]
    doc_lens = np.array(doc_lens)
    print('Doc lengths : AVG = {:.1f} , MED = {} , MIN = {:.1f} , MAX = {:.1f}'
          .format(doc_lens.mean(), np.median(doc_lens), doc_lens.min(), doc_lens.max()))

    q_lens = [len(d['stem']) for d in qa_dataset.data]
    q_lens = np.array(q_lens)
    print('Q lengths : AVG = {:.1f} , MED = {} , MIN = {:.1f} , MAX = {:.1f}'
          .format(q_lens.mean(), np.median(q_lens), q_lens.min(), q_lens.max()))

    c_lens = [max([len(c) for c in d['choices']]) for d in qa_dataset.data]
    c_lens = np.array(c_lens)
    print('A lengths : AVG = {:.1f} , MED = {} , MIN = {:.1f} , MAX = {:.1f}'
          .format(c_lens.mean(), np.median(c_lens), c_lens.min(), c_lens.max()))

    # Doc lengths : AVG = 1499.7 , MED = 1481.0 , MIN = 351.0 , MAX = 4137.0
    # Q lengths : AVG = 14.5 , MED = 14.0 , MIN = 7.0 , MAX = 29.0
    # A lengths : AVG = 8.5 , MED = 7.0 , MIN = 2.0 , MAX = 36.0
