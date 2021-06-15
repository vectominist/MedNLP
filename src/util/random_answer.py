'''
    File      [ src/util/random_answer.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ Generate random answers for the QA task ]
'''

import csv


def generate_qa_same(ids: int = 192, path: str = 'qa.csv'):
    with open(path, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'answer'])
        for i in range(ids):
            writer.writerow([str(i + 1), 'B'])


if __name__ == '__main__':
    generate_qa_same()
