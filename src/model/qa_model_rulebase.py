'''
    File      [ src/model/qa_model_rulebase_2.py ]
    Author    [ Chun-Wei Ho (NTUEE) ]
    Synopsis  [ Rule-based QA method ]
'''

import numpy as np
import tqdm
import edit_distance
import re
import multiprocessing as mp


def is_inv(sent: str):
    exp = '錯|誤|不|沒|(非(?!常|洲))'
    return bool(re.search(exp, sent)), re.sub(exp, '', sent)


def get_sim(sent: str, doc: list):
    match_sm = [edit_distance.SequenceMatcher(
        i, sent, action_function=edit_distance.highest_match_action) for i in doc]
    match_score = np.array([i.matches() for i in match_sm], dtype=np.float32)
    match_score -= match_score[match_score <=
                               np.percentile(match_score, 80)].mean()
    match_score[match_score < 0] = 0

    # _filter = [1,0.5,0.25]
    # match_score = np.convolve(match_score, _filter, 'full')[:-len(_filter) + 1]

    # sent_inv = is_inv(sent)
    # inv = [is_inv(i) ^ sent_inv for i in doc]
    # match_score[inv] *= -1

    # match_score /= len(sent)
    _filter = [1, 0.5, 0.25]
    match_score = np.convolve(match_score, _filter, 'full')[:-len(_filter) + 1]
    match_score[-1] += 1e-10

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
        (match_score / (np.array(match_len) + 1e-10)) ** 0.7
    match_score -= match_score.mean()

    _filter = [1, 0.6, 0.36]
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


class RuleBaseQA():
    def predict(self, dataset):
        with tqdm.tqdm(dataset) as prog_bar:
            with mp.Pool() as p:
                answers = p.map(self._predict_single_question, prog_bar)
        return np.array(answers)

    def _predict_single_question(self, question):
        doc = question['doc']
        stem = question['stem']
        choices = question['choices']

        stem = re.sub("下列|關於|何者|敘述|民眾|請問|正確|的|醫師", '', stem)
        inv, stem = is_inv(stem)
        choices = [re.sub('|'.join(stem) + '|民眾|醫師', '', i) for i in choices]
        if inv:
            ref_sim = get_sim(stem, doc)
            sim = [get_sim_with_inv(i, doc) for i in choices]
            score = np.corrcoef([ref_sim, *sim])[0, 1:]

            score2 = ((score + 1) / 2) ** 0.6 * np.max(sim, axis=1)
            return np.argmin(score2)
        else:
            ref_sim = get_sim(stem, doc)
            sim = [get_sim(i, doc) for i in choices]
            score = np.cov([ref_sim, *sim])[0, 1:]
            return np.argmax(score)


if __name__ == '__main__':
    pass
