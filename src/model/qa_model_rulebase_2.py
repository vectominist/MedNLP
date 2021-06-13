import numpy as np
import tqdm
import edit_distance
# from scipy import spatial
import re
import multiprocessing as mp

inv_chars = '錯|誤|有誤|不|沒|(非(?!常|洲))|(無(?!套))'


def is_inv(sent: str):
    return bool(re.search(inv_chars, sent)), re.sub(inv_chars, '', sent)


def invert_sentiment(sent: str):
    if bool(re.search(inv_chars, sent)):
        # negative
        if sent.find('沒有') >= 0:
            sent = sent.replace('沒有', '有')
        elif sent.find('不是') >= 0:
            sent = sent.replace('不是', '是')
        elif sent.find('不可能') >= 0:
            sent = sent.replace('不可能', '可能')
        sent = re.sub(inv_chars, '', sent)
    else:
        # positive
        if sent.find('有') >= 0:
            sent = sent.replace('有', '沒有')
        elif sent.find('是') >= 0:
            sent = sent.replace('是', '不是')
        elif sent.find('可能') >= 0:
            sent = sent.replace('可能', '不可能')
    return sent


def get_sim(sent: str, doc: list):
    match_sm = [edit_distance.SequenceMatcher(
        i, sent, action_function=edit_distance.highest_match_action) for i in doc]
    match_score = np.array([i.matches() for i in match_sm], dtype=np.float32)
    # print(match_score)

    match_score -= match_score[match_score <=
                               np.percentile(match_score, 80)].mean()
    match_score[match_score < 0] = 0
    # print(match_score)

    # _filter = [1,0.5,0.25]
    # match_score = np.convolve(match_score, _filter, 'full')[:-len(_filter) + 1]

    # sent_inv = is_inv(sent)
    # inv = [is_inv(i) ^ sent_inv for i in doc]
    # match_score[inv] *= -1

    # match_score /= len(sent)
    _filter = [1, 0.4, 0.4, 0.2]
    # _filter = [0.1, 0.2, 0.4, 0.2, 0.1]
    match_score = np.convolve(match_score, _filter, 'full')[:-len(_filter) + 1]
    match_score[-1] += 1e-10
    # print(match_score)

    return match_score


def get_sim_with_inv(sent: str, doc: list):
    match_sm = [edit_distance.SequenceMatcher(
        i, sent, action_function=edit_distance.highest_match_action) for i in doc]
    match_score = np.array([i.matches() for i in match_sm], dtype=np.float32)

    # match_seq = []
    match_len = []
    for sm, s in zip(match_sm, doc):
        blocks = [*sm.get_matching_blocks()]
        if len(blocks) == 0:
            # match_seq.append("")
            match_len.append(0)
        else:
            # match_seq.append(s[blocks[0][0]:blocks[-1][0] + 1])
            match_len.append(blocks[-1][0] - blocks[0][0] + 1)

    match_score = match_score * \
        (match_score / (np.array(match_len) + 1e-10)) ** 0.5
    match_score -= match_score.mean()

    _filter = [1, 0.5, 0.4, 0.1]
    match_score = np.convolve(match_score, _filter, 'full')[:-len(_filter) + 1]

    sent_inv = is_inv(sent)[0]
    inv = [is_inv(i)[0] ^ sent_inv for i in doc]
    match_score[inv] *= -1

    # match_score /= len(sent)
    # match_score -= match_score.mean()

    _filter = [1, 0.6, 0.36]
    match_score = np.convolve(match_score, _filter, 'full')[:-len(_filter) + 1]
    match_score[-1] += 1e-10

    return match_score


class RuleBaseQA2():
    def predict(self, dataset):
        with tqdm.tqdm(dataset) as prog_bar:
            with mp.Pool() as p:
                answers = p.map(self._predict_single_question, prog_bar)

        # answers = []
        # for d in dataset:
        #     print(d['doc'])
        #     print(d['stem'])
        #     print(d['choices'])

        #     res = self._predict_single_question(d)
        #     answers.append(res)
        #     print('answer =', d['answer'])
        #     print(res)
        #     input()

        scores = [a[0] for a in answers]
        is_inv = [a[1] for a in answers]
        return np.array(scores), np.array(is_inv, dtype=bool)

    def _predict_single_question(self, question):
        doc = question['doc']
        stem = question['stem']
        choices = question['choices']

        stem = re.sub("下列|關於|何者|敘述|民眾|請問|正確|的|醫師", '', stem)
        inv, stem = is_inv(stem)
        # choices = [re.sub('|'.join(stem) + '|民眾|醫師', '', i) for i in choices]
        # print('|'.join(stem))
        # choices = [re.sub('|'.join(stem) + '|民眾|醫師|的|覺得', '', i)
        #            for i in choices]
        choices = [re.sub('|民眾|醫師|的|覺得|這件事|這', '', i)
                   for i in choices]

        if inv:
            ref_sim = get_sim(stem, doc)
            sim = [get_sim_with_inv(i, doc) for i in choices]
            score = np.corrcoef([ref_sim, *sim])[0, 1:]

            score2 = ((score + 1) / 2) ** 0.6 * np.max(sim, axis=1)
            score2 = score
            return np.argmin(score2), True
        else:
            ref_sim = get_sim(stem, doc)
            sim = [get_sim(i, doc) for i in choices]
            score = np.cov([ref_sim, *sim])[0, 1:]
            return np.argmax(score), False


if __name__ == '__main__':
    pass
