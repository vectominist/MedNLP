import argparse
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import logging
logging.basicConfig(format='%(message)s', level=logging.INFO)


def read_corpus(path: str):
    with open(path, 'r') as fp:
        data = []
        for i, line in enumerate(fp):
            tokens = line.strip().split(' ')
            data.append(TaggedDocument(tokens, [i]))
        return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Doc2Vec training')
    parser.add_argument('--corpus', type=str, help='Path to text corpus')
    parser.add_argument('--save-path', type=str, help='Path to save model')
    args = parser.parse_args()

    model = Doc2Vec(
        vector_size=256,
        window=3,
        min_count=4,
        sample=1e-5,
        negative=5,
        epochs=10,
        workers=4,
        seed=7122)
    
    corpus = read_corpus(args.corpus)
    # model.build_vocab(corpus_file=args.corpus)
    model.build_vocab(corpus)
    model.train(
        corpus,
        total_examples=model.corpus_count,
        total_words=model.corpus_total_words,
        epochs=model.epochs)

    model.save(args.save_path)

    samples = [
        '所以 你現 在 都 是 約 的 你 有 固炮 了 還約',
        '所以 你 會 怕 你 現在 這樣子 就是 剛 治療 完 的 疾病 然 後 又 發生',
        '那 你 為 什麼 會 想 來 吃 因為 覺得 你 有 風險',
        '性行為 好 所以 你 是 很 怕 感染 hiv',
        '七夕 剛過 沒 關係 哈 好 那 你 以前 曾經 吃過 暴露 後 預防',
        '好 那 我們 反正 就是 hpv 反正 打 三支 現在 兩個 月 後 和 半年 後',
        '很濃 所以 有 發炎 這樣子 肚子 還會痛'
    ]
    for s in samples:
        print('Ref : {}'.format(s))
        inferred_vector = model.infer_vector(s.split(' '))
        sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
        print()
        for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
            print(u'%s %s: %s\n' % (label, sims[index], ' '.join(corpus[sims[index][0]].words)))
