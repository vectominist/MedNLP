import csv
import json


def read_csv(path):
    with open(path, 'r') as fp:
        rows = csv.reader(fp)
        data = []
        for i, r in enumerate(rows):
            if i == 0:
                continue
            data.append(
                (int(r[1]), r[2], int(r[3]))
            )
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
    train_csv = '../../data/Train_risk_classification_ans.csv'
    dev_csv = '../../data/Develop_risk_classification.csv'
    out_csv = '../../data/train_risk_tr-dv.csv'

    tr_set = read_csv(train_csv)
    dv_set = read_csv(dev_csv)
    merge_csv([tr_set, dv_set], out_csv)

    train_json = '../../data/Train_qa_ans.json'
    dev_json = '../../data/Develop_QA.json'
    out_json = '../../data/train_qa_tr-dv.json'

    tr_set = read_json(train_json)
    dv_set = read_json(dev_json)
    merge_json([tr_set, dv_set], out_json)