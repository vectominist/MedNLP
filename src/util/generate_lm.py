import argparse
import csv
import pickle
import json

from text_normalization import normalize_sent_with_jieba
from opencc import OpenCC

cc = OpenCC('s2t')

def read_risk_data(path: str):
    with open(path, 'r') as fp:
        sents = []
        rows = csv.reader(fp)
        for i, row in enumerate(rows):
            if i == 0:
                continue
            sent = row[2]
            sent = normalize_sent_with_jieba(sent)
            sents += sent
        return sents


def read_MedDG_data(path: str):
    with open(path,'rb') as fp:
        data = pickle.load(fp)
        sents = []
        for d in data:
            sent = ''
            for turn in d:
                if len(turn['Sentence']) > 0:
                    if turn['id'] == 'Patients':
                        sent += '民眾：'
                    elif turn['id'] == 'Doctor':
                        sent += '醫師：'
                    else:
                        raise ValueError(turn['id'])
                    sent += cc.convert(turn['Sentence'])
            sent = normalize_sent_with_jieba(sent)
            sents += sent
        return sents


def read_MedicalDialogue_data(path: str):
    with open(path, 'r') as fp:
        data = json.load(fp)
        sents = []
        for d in data:
            # d: list
            for sent in d:
                sent = normalize_sent_with_jieba(sent)
                sents += cc.convert(sent)
        return sents


def write_to_txt(data: list, path: str):
    with open(path, 'w') as fp:
        for d in data:
            fp.write(' '.join(d) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LM data generator for Doc2Vec')
    parser.add_argument('--risk-path', type=str, help='Path to risk data (.csv)')
    parser.add_argument('--meddg-path', type=str, help='Path to MedDG data (.pk)')
    parser.add_argument('--meddia-path', type=str, help='Path to Medical Dialogue data (.json)')
    parser.add_argument('--out-path', type=str, help='Path to output .txt file')
    args = parser.parse_args()

    risk_utts = read_risk_data(args.risk_path)
    print('Found {} utterances from {}'.format(len(risk_utts), args.risk_path))
    meddg_utts = read_MedDG_data(args.meddg_path)
    print('Found {} utterances from {}'.format(len(meddg_utts), args.meddg_path))
    meddia_utts = read_MedicalDialogue_data(args.meddia_path)
    print('Found {} utterances from {}'.format(len(meddia_utts), args.meddia_path))
    all_utts = risk_utts + meddg_utts + meddia_utts
    print('Total {} utterances found'.format(len(all_utts)))
    write_to_txt(all_utts, args.out_path)
    print('Generated data for LM and saved to {}'.format(args.out_path))