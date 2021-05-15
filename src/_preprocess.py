import numpy as np
import csv
import json
import unicodedata
import re
import time
import random
import os
from tqdm import tqdm
from dataset import preprocess, split_sent

'''
Here we will convert the word vector into a readable form, and generate two file and a path: 
1. vocab.json     : Used to record the index of each text
2. embeddings.npy : Used to record the word vector in npy form
3. ./data         : Used to store the generated data, and our dataset in here
'''

def load_vectors(fname):
    if os.path.isdir("./data"):
        pass
    else:
        os.makedirs("./data")

    with open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as WordEmbeddings,\
            open("data/vocab.json", 'w', encoding='utf-8') as vocab_write:
        wv = []
        vocab = {}
        vocab["[PAD]"] = 0
        wv.append(np.zeros(300))
        for i, line in enumerate(WordEmbeddings):
            word, vec = line.split(' ', 1)
            if i == 0:
                continue
            vocab[word] = i
            wv.append(np.fromstring(vec, sep=' '))
        json.dump(vocab, vocab_write, ensure_ascii=False)

    np.save("data/embeddings.npy", np.array(wv))

'''
Put your word vector path here.
In this baseline, our word vector is to use fasttext.
You can find it in here. (https://fasttext.cc/docs/en/crawl-vectors.html)
'''

if __name__ == "__main__":
    load_vectors("word2vec/cc.zh.300.vec")
