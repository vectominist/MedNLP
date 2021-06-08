import numpy as np
import tqdm
import edit_distance
import re

def is_inv(sent:str):
    return bool(re.search('錯|誤|不|沒|(非(?!常|洲))',sent))
def get_sim(sent:str, doc:list):
    sim = np.array([edit_distance.SequenceMatcher(i, sent, action_function=edit_distance.highest_match_action).matches() \
                                 for i in doc], dtype=np.float32)

    sim -= sim[sim <= np.percentile(sim, 80)].mean()
    sim[sim < 0] = 0

    # _filter = [1,0.5,0.25]
    # sim = np.convolve(sim, _filter, 'full')[:-len(_filter) + 1]
    
    # sent_inv = is_inv(sent)
    # inv = [is_inv(i) ^ sent_inv for i in doc]
    # sim[inv] *= -1

    # sim /= len(sent)
    _filter = [1, 0.5, 0.25]
    sim = np.convolve(sim, _filter, 'full')[:-len(_filter) + 1]
    
    return sim
def get_correct(sent:str, doc:list):
    correct = np.array([edit_distance.SequenceMatcher(i, sent, action_function=edit_distance.highest_match_action).matches() \
                                 for i in doc], dtype=np.float32) / len(sent)

    correct -= correct[correct <= np.percentile(correct, 80)].mean()
    correct[correct < 0] = 0
    
    _filter = [1, 0.6, 0.36]
    correct = np.convolve(correct, _filter, 'full')[:-len(_filter) + 1]

    sent_inv = is_inv(sent)
    inv = [is_inv(i) ^ sent_inv for i in doc]
    correct[inv] *= -1

    # correct /= len(sent)
    # correct -= correct.mean()

    _filter = [1, 0.6, 0.36]
    correct = np.convolve(correct, _filter, 'full')[:-len(_filter) + 1]

    return correct.max()

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
