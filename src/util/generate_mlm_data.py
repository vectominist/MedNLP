def read_data(path):
    with open(path, 'r') as fp:
        data = []
        for line in fp.readlines():
            line = line.strip()
            if line == '':
                continue
            data.append(line)
        return data


def write_data(data, path, min_len=100, max_len=510):
    with open(path, 'w') as fp:
        current_seq = []
        current_len = 0
        for s in data:
            while current_len + len(s) > max_len:
                current_len -= len(current_seq[0])
                current_seq = current_seq[1:]
            current_seq.append(s)
            current_len += len(s)
            fp.write(s + '\n')
            if current_len >= min_len:
                fp.write(' '.join(current_seq) + '\n')
                fp.write(
                    ' '.join(current_seq[:len(current_seq) // 2]) + '\n')
                fp.write(
                    ' '.join(current_seq[len(current_seq) // 2:]) + '\n')


if __name__ == '__main__':
    orig_path = '../../data/mlm_qa_tr-dv.txt'
    target_path = '../../data/mlm_qa_tr-dv_doc.txt'

    data = read_data(orig_path)
    write_data(data, target_path, 100, 510)
