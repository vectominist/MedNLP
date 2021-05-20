import csv

def generate_qa_same(ids: int = 192, path: str = 'qa.csv'):
    with open(path, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'answer'])
        for i in range(ids):
            writer.writerow([str(i + 1), 'A'])


if __name__ == '__main__':
    generate_qa_same()
