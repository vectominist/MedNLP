import numpy as np
import tqdm
import edit_distance
import re

def is_inv(sent:str):
    return bool(re.search('錯|誤|不|沒|(非(?!常|洲))',sent))

def get_sim(sent:str, doc:list):
    match_sm = [edit_distance.SequenceMatcher(i, sent, action_function=edit_distance.highest_match_action) for i in doc]
    match_score = np.array([i.matches() for i in match_sm], dtype=np.float32)
    match_score -= match_score[match_score <= np.percentile(match_score, 80)].mean()
    match_score[match_score < 0] = 0

    # _filter = [1,0.5,0.25]
    # match_score = np.convolve(match_score, _filter, 'full')[:-len(_filter) + 1]
    
    # sent_inv = is_inv(sent)
    # inv = [is_inv(i) ^ sent_inv for i in doc]
    # match_score[inv] *= -1

    # match_score /= len(sent)
    _filter = [1, 0.5, 0.25]
    match_score = np.convolve(match_score, _filter, 'full')[:-len(_filter) + 1]
    
    return match_score

def get_correct(sent:str, doc:list):
    match_sm = [edit_distance.SequenceMatcher(i, sent, action_function=edit_distance.highest_match_action) for i in doc]
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

    match_score = match_score ** 2 / (np.array(match_len) + 1e-10)
    match_score -= match_score.mean()

    _filter = [1, 0.6, 0.36]
    match_score = np.convolve(match_score, _filter, 'full')[:-len(_filter) + 1]

    sent_inv = is_inv(sent)
    inv = [is_inv(i) ^ sent_inv for i in doc]
    match_score[inv] *= -1

    # match_score /= len(sent)
    # match_score -= match_score.mean()

    _filter = [1, 0.6, 0.36]
    match_score = np.convolve(match_score, _filter, 'full')[:-len(_filter) + 1]

    return match_score.max()

class RuleBaseQA():
    def predict(self, dataset):
        answers = []
        with tqdm.tqdm(dataset) as prog_bar:
            for question in prog_bar:
                doc = question['doc']
                stem = question['stem']
                choices = question['choices']
                
                stem = re.sub("下列|關於|何者|敘述|民眾",'',stem)

                if is_inv(stem):
                    correct = [get_correct(i,doc) for i in choices]
                    answers.append(np.argmin(correct))
                else:
                    ref_sim = get_sim(stem, doc)
                    sim = [get_sim(i, doc) for i in choices]
                    score = np.cov([ref_sim, *sim])[0,1:]
                    answers.append(np.argmax(score))
        return np.array(answers)

if __name__ == '__main__':
    pass
