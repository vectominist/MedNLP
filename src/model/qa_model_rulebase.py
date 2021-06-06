import numpy as np
import tqdm

class RuleBaseQA():
    def predict(self, dataset):
        answer = []
        with tqdm.tqdm(dataset) as prog_bar:
            for i in prog_bar:
                doc = i['doc']
                stem = i['stem']
                choices = i['choices']
                answer.append(0)
        return np.array(answer)

if __name__ == '__main__':
    pass
