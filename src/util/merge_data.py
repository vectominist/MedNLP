'''
    File      [ src/util/merge_data.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ Merge train & labeled dev sets ]
'''

import csv
import json


def read_csv(path):
    with open(path, 'r') as fp:
        rows = csv.reader(fp)
        data = []
        for i, r in enumerate(rows):
            if i == 0:
                continue
            id1, id2 = int(r[1]), r[2]
            label = 0 if len(r) == 3 else int(r[3])
            data.append((id1, id2, label))
        return data


def merge_csv(data_list, path):
    with open(path, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['', 'article_id', 'text', 'label'])
        idx = 0
        for data in data_list:
            for i, d in enumerate(data):
                writer.writerow([idx + 1, idx + 1, d[1], d[2]])
                idx += 1


def read_json(path):
    with open(path, 'r', encoding='UTF-8') as fp:
        data = json.load(fp)
        return data


def merge_json(data_list, path):
    with open(path, 'w', encoding='UTF-8') as fp:
        prev_article_id = 0
        idx, article_id = 0, 0
        all_data_list = []
        for data in data_list:
            for d in data:
                if d['article_id'] != prev_article_id:
                    prev_article_id = int(d['article_id'])
                    article_id += 1
                idx += 1
                d['id'] = idx
                d['article_id'] = article_id
                all_data_list.append(d)
        json.dump(all_data_list, fp, ensure_ascii=False)


if __name__ == '__main__':
    train_csv = 'data/Train_risk_classification_ans.csv'
    dev_csv = 'data/Develop_risk_classification_ans.csv'
    test_csv = 'data/Test_risk_classification.csv'
    out_csv = 'data/train_risk_tr-dv-tt.csv'

    tr_set = read_csv(train_csv)
    dv_set = read_csv(dev_csv)
    tt_set = read_csv(test_csv)
    all_set = [tr_set, dv_set, tt_set]
    tot_len = len(tr_set) + len(dv_set) + len(tt_set)
    print('CSV: Total number of samples = {}'.format(tot_len))
    merge_csv(all_set, out_csv)

    # train_json = 'data/Train_qa_ans.json'
    # dev_json = 'data/Develop_QA.json'
    # out_json = 'data/train_qa_tr-dv.json'

    # tr_set = read_json(train_json)
    # dv_set = read_json(dev_json)
    # merge_json([tr_set, dv_set], out_json)
