import csv
import json
from torch.utils.data import Dataset
import unicodedata
from dataset import split_sent
import tqdm
import numpy as np

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
tokenize = lambda x,max_length: tokenizer(x, return_tensors="pt", padding="max_length", truncation="longest_first", max_length=max_length).input_ids

# TODO: Text normalization


choice2int = {
    'A': 0, 'B': 1, 'C': 2,
    'Ａ': 0, 'Ｂ': 1, 'Ｃ': 2
}


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
                if split in ['train', 'val']:
                    idx, sent, label = int(row[1]), row[2], int(row[3])
                    data.append((idx, sent, label))
                else:
                    idx, sent = int(row[1]), row[2]
                    data.append((idx, sent))

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
        return self.data[index]


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
                        answer = [k for k in range(len(choices)) if choices[k] == d['answer']][0]
                    else:
                        answer = choice2int[d['answer']]

                text = unicodedata.normalize("NFKC", text)
                text = ["".join(i) for i in split_sent(text)]
                text = [""] * max(0, max_doc_len - len(text)) + text[:max_doc_len]
                stem = unicodedata.normalize("NFKC", stem)
                choices = [unicodedata.normalize("NFKC", i) for i in choices]

                text = tokenize(text, max_sent_len)
                stem = tokenize(stem, max_q_len).squeeze()
                choices = tokenize(choices, max_c_len)

                if split in ['train', 'val']:
                    one_hot_answer = np.zeros((3,), dtype = np.float32)
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
        # '/Users/hc/Downloads/Train_risk_classification_ans.csv', 'train')
    # print(cl_dataset.__getitem__(0))

    qa_dataset = QADataset(
        'data/Train_qa_ans.json', 'train')
    # print(qa_dataset.__getitem__(0))
    # print(qa_dataset[0])
