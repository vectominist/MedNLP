'''
    File      [ src/util/lm_normalizer.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ Normalization of LM data ]
'''

from opencc import OpenCC
import re
cc = OpenCC('s2t')


def is_mandarin(c: str) -> bool:
    return len(re.findall(r'[\u4e00-\u9fff]+', c)) > 0


def merge_chinese(text: str) -> str:
    if len(text) <= 2:
        return text
    out_text = text[0]
    for i in range(1, len(text) - 1):
        if text[i] == ' ' and \
                (is_mandarin(text[i - 1]) or
                 is_mandarin(text[i + 1])):
            continue
        out_text += text[i]
    out_text += text[-1]
    return out_text


if __name__ == '__main__':
    fname = 'doc2vec_data_3.txt'
    target = 'doc2vec_data_3.norm.txt'
    with open(fname, 'r') as fin:
        with open(target, 'w') as fout:
            for line in fin.readlines():
                line = line.strip()
                if line == '':
                    continue
                line = merge_chinese(line)
                line = cc.convert(line)
                fout.write(line + '\n')
