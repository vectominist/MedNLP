'''
Datasets
'''

import csv
import json
import torch
# from torch.utils.data import Dataset
from torch.utils.data.dataset import Dataset
import unicodedata
from dataset import split_sent
import tqdm
import numpy as np

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")


def tokenize(x, max_length):
    return tokenizer(
        x, return_tensors="pt", padding="max_length",
        truncation="longest_first", max_length=max_length).input_ids


choice2int = {
    'A': 0, 'B': 1, 'C': 2,
    'Ａ': 0, 'Ｂ': 1, 'Ｃ': 2
}


def normalize_and_tokenize(text, max_doc_len=170, max_sent_len=70):
    text = unicodedata.normalize("NFKC", text)
    text = ["".join(i) for i in split_sent(text)]
    text = text[:max_doc_len] + [""] * max(0, max_doc_len - len(text))
    text = tokenizer(
        text, return_tensors="pt", padding="max_length",
        truncation="longest_first", max_length=max_sent_len)

    return text


class ClassificationDataset(Dataset):
    '''
        Dataset for classification
    '''

    def __init__(self, path, split='train', val_r=10):
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

        # import pdb
        # pdb.set_trace()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # train & val : idx, paragraph, label
        # dev & test : idx, paragraph
        if self.split in ['train', 'val']:
            item = {key: val[0] for key, val in self.data[index][1].items()}
            item['labels'] = torch.tensor(self.data[index][2])
            return item
        else:
            item = {key: val[0] for key, val in self.data[index][1].items()}
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


if __name__ == '__main__':
    # debug code
    # cl_dataset = ClassificationDataset(
    # 'data/Train_risk_classification_ans.csv', 'train')
    # print(cl_dataset.[0])

    qa_dataset = QADataset(
        'data/Train_qa_ans.json', 'train')
    # print(qa_dataset.__getitem__(0))
    # print(qa_dataset[0])
