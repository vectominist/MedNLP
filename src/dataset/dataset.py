import numpy as np
import csv
import json
import unicodedata
import re
from torch.utils.data import Dataset, DataLoader

'''
Here we will do preprocessing on the dataset.
Something needs to be done here :
1. Read the file in.
2. Separate the article, question, answer.
3. Used PAD to round each sentence into the same length
'''

def split_sent(sentence: str):
    first_role_idx = re.search(':', sentence).end(0)
    out = [sentence[:first_role_idx]]

    tmp = sentence[first_role_idx:]
    while tmp:
        res = re.search(
            r'(護理師[\w*]\s*:|醫師\s*:|民眾\s*:|家屬[\w*]\s*:|個管師\s*:)', tmp)
        if res is None:
            break

        idx = res.start(0)
        idx_end = res.end(0)
        out[-1] = list(out[-1] + tmp[:idx])
        out.append(tmp[idx:idx_end])
        tmp = tmp[idx_end:]

    out[-1] = list(out[-1] + tmp)

    return out

def preprocess(qa_file: str, risk_file: str):
    with open(qa_file, "r", encoding = "utf-8") as f_QA,\
            open(risk_file, "r", encoding = "utf-8") as f_Risk:
        article = []
        risk = []
        
        # One smaple of Article
        # [[Sent_1], [Sent_2], ..., [Sent_n]]
        for i, line in enumerate(csv.reader(f_Risk)):
            if i == 0:
                continue
            text = unicodedata.normalize("NFKC", line[2])
            text = text.replace(" ", "")
            article.append(split_sent(text))
            risk.append(int(line[3]))

        # One sample of QA
        # [Question, [[Choice_1, Answer_1], [Choice_2, Answer_2], [Choice_3, Answer_3]]]
        article_id = 1
        qa = []
        qa.append([])
        for data in json.load(f_QA):
            question = data["question"]
            temp = []
            answer = ""
            for choice in question["choices"]:
                text = list(unicodedata.normalize("NFKC", choice["text"]))
                if unicodedata.normalize("NFKC", choice["label"]) in unicodedata.normalize("NFKC",data["answer"]):
                    temp.append([text, 1])
                    answer = data["answer"]
                else:
                    temp.append([text, 0])
            question_text = list(
                unicodedata.normalize("NFKC", question["stem"]))
            if not answer:
                print("".join(question_text))
                print(question["choices"])
                print(unicodedata.normalize("NFKC",data["answer"]))
                print(question["choices"][2]["label"])
            if data["article_id"] != article_id:
                qa.append([])
                article_id = data["article_id"]

            qa[-1].append([question_text, temp])

    return article, risk, qa


def encode_sent(w2id: dict, sentence: list, max_length: int):
    output = []
    for i, token in enumerate(sentence):
        if i >= max_length:
            break
        if token in w2id:
            output.append(w2id[token])
        else:
            output.append(0)
    padding_word = [0]
    sent_padding_size = max_length - len(output)
    output = output + padding_word*sent_padding_size

    return output

class all_dataset(Dataset):
    def __init__(
        self,
        qa_file: str,
        risk_file: str,
        max_sent_len: int = 52,
        max_doc_len: int = 170,
        max_q_len: int = 20,
        max_c_len: int = 18
    ):
        super().__init__()
        with open('data/vocab.json', 'r', encoding='utf-8') as f_w2id:
            w2id = json.load(f_w2id)
        # w2id = {"[PAD]": 0}
        article, risk, qa = preprocess(qa_file, risk_file)

        # `risk` shape: [N]
        self.risk = np.array(risk)

        # `article` shape: [N, `max_doc_len`, `max_sent_len`]
        self.article = []
        for document in article:
            self.article.append([])
            for i, sentence in enumerate(document):
                if i >= max_doc_len:
                    break
                self.article[-1].append([])
                self.article[-1][-1] = encode_sent(w2id,
                                                   sentence, max_sent_len)
            padding_sent = [[0]*max_sent_len]
            doc_padding_size = max_doc_len - len(self.article[-1])
            self.article[-1] = self.article[-1] + padding_sent*doc_padding_size

        self.article = np.array(self.article)
        self.QA = []
        
        for idx, article in enumerate(qa):
            for question_data in article:
                q_text = encode_sent(w2id, question_data[0], max_q_len)
                temp = {}
                temp["article_id"] = idx
                temp["question"] = np.array(q_text)
                temp["choice"] = []
                temp["qa_answer"] = []
                for choice_data in question_data[1]:
                    c_text = encode_sent(w2id, choice_data[0], max_c_len)
                    temp["choice"].append(c_text)
                    temp["qa_answer"].append(choice_data[1])
                temp["choice"] = np.array(temp["choice"])
                temp["qa_answer"] = np.array(temp["qa_answer"])
                self.QA.append(temp)

    def __len__(self):
        return len(self.QA)

    def __getitem__(self, idx: int):
        return self.QA[idx]