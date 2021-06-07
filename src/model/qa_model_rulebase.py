import numpy as np
import tqdm
import edit_distance
from scipy.stats import kurtosis

def is_inv(sent:str):
    for i in ["錯誤","有誤","不","非","沒"]:
        if i in sent:
            return True
    return False
def get_sim(sent:str, doc:list):
    sim = np.array([edit_distance.SequenceMatcher(i, sent).matches() for i in doc], dtype=np.float32)

    # _filter = [1,0.5,0.25]
    # sim = np.convolve(sim, _filter, 'full')[:-len(_filter) + 1]
    
    # inv = [is_inv(i) for i in doc]
    # sim[inv] *= -1
    # if is_inv(sent):
    #     sim *= -1

    # sim /= len(sent)
    _filter = [1,0.5,0.25]
    sim = np.convolve(sim, _filter, 'full')[:-len(_filter) + 1]
    return sim
def get_correct(sent:str, doc:list):
    correct = np.array([edit_distance.SequenceMatcher(i, sent).matches() for i in doc], dtype=np.float32)

    _filter = [1,0.5,0.25]
    correct = np.convolve(correct, _filter, 'full')[:-len(_filter) + 1]

    inv = [is_inv(i) for i in doc]
    correct[inv] *= -1
    if is_inv(sent):
        correct *= -1

    # correct -= correct.mean()
    # correct /= len(sent)
    _filter = [1,0.5,0.25]
    correct = np.convolve(correct, _filter, 'full')[:-len(_filter) + 1]

    return correct.max()

class RuleBaseQA():
    def predict(self, dataset):
        answers = []
        with tqdm.tqdm(dataset) as prog_bar:
            for i in prog_bar:
                doc = i['doc']
                stem = i['stem']
                choices = i['choices']
                if is_inv(stem):
                    correct = [get_correct(i,doc) for i in choices]
                    answers.append(np.argmin(correct))
                else:
                    ref_sim = get_sim(stem, doc)
                    sim = [get_sim(i, doc) for i in choices]
                    score = np.cov([ref_sim, *sim])[0,1:]
                    answers.append(score.argmax())
        return np.array(answers)

if __name__ == '__main__':
    pass
