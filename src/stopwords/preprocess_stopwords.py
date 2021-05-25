import unicodedata
import re
import string
from opencc import OpenCC

cc = OpenCC('s2t')

chinese_punctuations = r"""，。、！？‘’·⋯》”“"""
remove_punctuations = string.punctuation + chinese_punctuations
translator = str.maketrans(remove_punctuations, ' ' * len(remove_punctuations))


def collect_stopwords(path: str, begin_idx: int = 0) -> set:
    with open(path, 'r') as fp:
        words = set()
        i = 0
        for w in fp.readlines():
            if i < begin_idx:
                i += 1
                continue
            w = w.strip()
            w = unicodedata.normalize("NFKC", w).lower()
            w = w.translate(translator)
            w = w.replace('.', '')
            w = w.replace(' ', '')
            if w == '' or \
                    (ord(w[0]) >= ord('a') and ord(w[0]) <= ord('z')):
                continue
            words.add(cc.convert(w))
        return words


def write_to_txt(stopwords_set: set, path: str):
    with open(path, 'w') as fp:
        stopwords = list(stopwords_set)
        stopwords = sorted(stopwords)
        for w in stopwords:
            fp.write(w + '\n')


if __name__ == '__main__':
    stopword_list = [
        'baidu_stopwords.txt',
        'cn_stopwords.txt',
        'hit_stopwords.txt',
        'scu_stopwords.txt'
    ]
    begin_idx = [
        6, 18, 263, 0
    ]

    stopwords_set = set()
    for p, idx in zip(stopword_list, begin_idx):
        words = collect_stopwords(p, idx)
        stopwords_set = set.union(stopwords_set, words)

    write_to_txt(stopwords_set, 'processed_stopwords.txt')
    # should find 1495 stop words
