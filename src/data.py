'''
Datasets
'''

from transformers import AutoTokenizer, BertTokenizerFast
import csv
import json
import torch
from torch.utils.data.dataset import Dataset
import unicodedata
import re
from dataset import split_sent
import tqdm
import numpy as np
from augmentation import sentence_random_removal
from opencc import OpenCC

cc = OpenCC('s2t')  # simplified to traditional

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
# tokenizer_risk = AutoTokenizer.from_pretrained(
#     "sentence-transformers/stsb-xlm-r-multilingual")
# tokenizer_risk = AutoTokenizer.from_pretrained(
#     "DeepPavlov/bert-base-multilingual-cased-sentence")
# tokenizer_risk = AutoTokenizer.from_pretrained(
#     "sentence-transformers/distilbert-multilingual-nli-stsb-quora-ranking")
tokenizer_risk = BertTokenizerFast.from_pretrained('ckiplab/bert-base-chinese')

choice2int = {
    'A': 0, 'B': 1, 'C': 2,
    'Ａ': 0, 'Ｂ': 1, 'Ｃ': 2
}


def tokenize(x, max_length):
    return tokenizer(
        x, return_tensors="pt", padding="max_length",
        truncation="longest_first", max_length=max_length).input_ids


def is_mandarin(c: str) -> bool:
    return len(re.findall(r'[\u4e00-\u9fff]+', c)) > 0


def split_chinese(text: str) -> str:
    if len(text) <= 2:
        return text
    out_text = text[0]
    for i in range(1, len(text) - 1):
        if text[i] == ' ' and \
                (is_mandarin(text[i - 1]) or
                 is_mandarin(text[i + 1])):
            continue
        out_text += text[i]
    return out_text


def normalize_and_tokenize(text, max_doc_len=120, max_sent_len=50):
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace('.', '')
    text = text.replace(' ', '')
    text = ["".join(i) for i in split_sent(text)]
    text = text[:max_doc_len] + [""] * max(0, max_doc_len - len(text))
    text = tokenizer_risk(
        text, return_tensors="pt", padding="max_length",
        truncation="longest_first", max_length=max_sent_len)

    return text


class ClassificationDataset(Dataset):
    '''
        Dataset for classification
    '''

    def __init__(self, path, split='train', val_r=10, rand_remove=False):
        assert split in ['train', 'val', 'dev', 'test']

        self.path = path
        self.split = split

        with open(path, 'r') as fp:
            data = []
            rows = csv.reader(fp)
            for i, row in enumerate(rows):
                if i == 0:
                    continue
                idx, sent = int(row[1]), row[2]
                sent = normalize_and_tokenize(sent)
                if split in ['train', 'val']:
                    label = int(row[3])
                    data.append((idx, sent, label))
                else:
                    data.append((idx, sent))

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
            item = {key: val for key, val in self.data[index][1].items()}
            if self.rand_remove:
                item = sentence_random_removal(item)
            item['labels'] = torch.tensor(self.data[index][2])
            return item
        else:
            item = {key: val for key, val in self.data[index][1].items()}
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

    def __init__(self, path, split='train', val_r=10, rand_remove=False):
        assert split in ['train', 'val', 'dev', 'test']

        self.path = path
        self.split = split

        with open(path, 'r') as fp:
            data = []
            for line in fp.readlines():
                line = line.strip()  # should be normalized and segmented
                if line == '':
                    continue
                line = split_chinese(line)
                line = cc.convert(line)
                line = tokenizer_risk(
                    [line], return_tensors="pt", padding="max_length",
                    truncation="longest_first", max_length=50)
                data.append(line)

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
        # train & val : idx, paragraph, label
        # dev & test : idx, paragraph
        if self.split in ['train', 'val']:
            item = {key: val for key, val in self.data[index][1].items()}
            return item
        else:
            item = {key: val for key, val in self.data[index][1].items()}
            return item


if __name__ == '__main__':
    # debug code
    # cl_dataset = ClassificationDataset(
    # 'data/Train_risk_classification_ans.csv', 'train')
    # print(cl_dataset.[0])

    qa_dataset = QADataset(
        'data/Train_qa_ans.json', 'train')
    # print(qa_dataset.__getitem__(0))
    # print(qa_dataset[0])
