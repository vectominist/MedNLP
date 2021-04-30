import csv
import json
from torch.utils.data import Dataset


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

    def __init__(self, path, split='train', val_r=10):
        assert split in ['train', 'val', 'dev', 'test']

        self.path = path
        self.split = split

        with open(path, 'r') as fp:
            data_list = json.load(fp)
            assert type(data_list) == list

            data = []
            for i, d in enumerate(data_list):
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

                if split in ['train', 'val']:
                    data.append((idx, text, stem, choices, answer))
                else:
                    data.append((idx, text, stem, choices))

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
    cl_dataset = ClassificationDataset(
        '/Users/hc/Downloads/Train_risk_classification_ans.csv', 'train')
    # print(cl_dataset.__getitem__(0))

    qa_dataset = QADataset(
        '/Users/hc/Downloads/Train_qa_ans.json', 'train')
    # print(qa_dataset.__getitem__(0))
