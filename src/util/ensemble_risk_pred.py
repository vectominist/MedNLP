import csv
import numpy as np


def read_csv(path):
    with open(path, 'r') as fp:
        rows = csv.reader(fp)
        ids, data = [], []
        for i, r in enumerate(rows):
            if i == 0:
                continue
            ids.append(r[0])
            data.append(float(r[1]))
        return ids, np.array(data)


def ensemble(data_list):
    # data_list: list of lists
    final_data = np.zeros((len(data_list[0]), ))
    for data in data_list:
        final_data += data
    final_data /= len(data_list)
    return final_data


def write_result(path, ids, data):
    with open(path, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['article_id', 'probability'])
        assert len(ids) == len(data)
        for i in range(len(data)):
            writer.writerow([str(ids[i]), data[i]])


if __name__ == '__main__':
    files = [
        'data/result_20210611-3/decision.csv',
        'data/result_20210613-1-gradboost-tr-dv/decision.csv'
    ]
    data_list = [read_csv(f) for f in files]
    ids = data_list[0][0]
    data_list = [d[1] for d in data_list]
    data = ensemble(data_list)
    write_result('data/decision.csv', ids, data)
