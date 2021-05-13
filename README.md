MedNLP
==============

enviroment
----------
- python 3.8
- torch 1.8

Preprocess
----------
- Download pre-trained word embedding to ``word2vec/cc.zh.300.vec``: [fasttext](https://fasttext.cc/docs/en/crawl-vectors.html)
- Run ``srd/_preprocess.py`` to generate two file. (``data/vocab.json`` and ``data/embeddings.npy``)
- Put your all dataset in ``data`` folder. (Run ``download.sh``)

Run
---
- Training and testing operations of Risk task are in ``src/_risk.py``
- Training and testing operations of QA task are in ``src/_qa.py``

Model
-----
- Hierarchical Attention Networks (see [paper](https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf) for more detial)
