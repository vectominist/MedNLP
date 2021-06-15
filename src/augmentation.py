'''
    File      [ src/augmentation.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ Text data augmentation ]
'''

import torch
import numpy as np
# import jieba
# import synonyms


with open('src/stopwords/processed_stopwords.txt', 'r') as fp:
    stopwords = [w.strip() for w in fp.readlines() if w.strip() != '']


def EDA(text: str,
        p_insert: float = 0.1,
        p_swap: float = 0.1,
        p_delete: float = 0.1) -> str:
    '''
        Easy Data Augmentation
    '''

    # text = eda_synonym_replacement(text)
    text = eda_random_insertion(text, p_insert)
    text = eda_random_swap(text, p_swap)
    text = eda_random_deletion(text, p_delete)

    return text


def eda_synonym_replacement(
    text: str,
    ratio: float = 0.1
) -> str:
    '''
        Implementation of EDA SR
        Warning: Using jieba + synonyms is very slow.
                 Should not be used during training.
    '''

    seg = (','.join(jieba.cut(text))).split(',')
    if int(len(seg) * ratio) <= 1:
        return text

    replace_n_word = np.random.randint(1, int(len(seg) * ratio))
    indices = np.random.permutation(len(seg))[:replace_n_word]
    for i in indices:
        cands = synonyms.nearby(seg[i], 2)[0]
        if len(cands) == 0:
            continue
        seg[i] = cands[1]

    return ''.join(seg)


def eda_random_insertion(
    text: str,
    ratio: float = 0.1
) -> str:
    '''
        Implementation of EDA RI
    '''

    if int(len(text) * ratio) < 1:
        return text

    insert_n_char = int(len(text) * ratio)
    indices = np.random.permutation(len(text))[:insert_n_char]
    for i in indices:
        idx = np.random.randint(0, len(stopwords))
        text = text[:i] + stopwords[idx] + text[i:]

    return text


def eda_random_swap(
    text: str,
    ratio: float = 0.1
) -> str:
    '''
        Implementation of EDA RS
    '''

    if int(len(text) * ratio) < 1:
        return text

    l = len(text)
    swap_n_char = int(l * ratio)
    text = list(text)
    for i in range(swap_n_char):
        swap_indices = np.random.permutation(l)[:2]
        text[swap_indices[0]], text[swap_indices[1]] = \
            text[swap_indices[1]], text[swap_indices[0]]

    return ''.join(text)


def eda_random_deletion(
    text: str,
    ratio: float = 0.1
) -> str:
    '''
        Implementation of EDA RD
    '''

    if int(len(text) * ratio) < 1:
        return text

    delete_n_char = int(len(text) * ratio)
    indices = np.random.permutation(len(text))[:-delete_n_char]
    indices = np.sort(indices)

    return ''.join([text[i] for i in indices])


def sentence_random_swap(
        input_dict: dict,
        max_swap_ratio: float = 0.1) -> dict:
    '''
        Randomly swap some of the sentences in a document
        input_dict: {
            'input_ids': torch.FloatTensor (Sentences x Length),
            'attention_mask': torch.LongTensor (Sentences x Length),
            (the elements should all be in the same shape)
        }
    '''

    n_sent = (input_dict['attention_mask'].sum(1) > 2).long().sum().item()
    if int(n_sent * max_swap_ratio) < 1:
        return input_dict
    swap_n_sent = int(n_sent * max_swap_ratio)
    output_dict = {}
    for i in range(swap_n_sent):
        swap_indices = np.random.permutation(n_sent)[:2]
        orig_indices = np.arange(input_dict['attention_mask'].shape[0])
        orig_indices[swap_indices[0]] = swap_indices[1]
        orig_indices[swap_indices[1]] = swap_indices[0]
        for key, val in input_dict.items():
            output_dict[key] = val[orig_indices, :]
    return input_dict


def sentence_random_removal(
        input_dict: dict,
        max_removal_ratio: float = 0.1) -> dict:
    '''
        Randomly removing some of the sentences in a document
        input_dict: {
            'input_ids': torch.FloatTensor (Sentences x Length),
            'attention_mask': torch.LongTensor (Sentences x Length),
            (the elements should all be in the same shape)
        }
    '''

    n_sent = input_dict['input_ids'].shape[0]
    if int(n_sent * max_removal_ratio) < 1:
        return input_dict
    remove_n_sent = int(n_sent * max_removal_ratio)
    indices = np.random.permutation(n_sent)[:-remove_n_sent]
    indices = np.sort(indices)
    output_dict = {}
    for key, val in input_dict.items():
        output_dict[key] = val[indices, :]
        if len(indices) < n_sent:
            output_dict[key] = torch.cat(
                [output_dict[key]] + [val[-1:, :]] * (n_sent - len(indices)), dim=0)
    return output_dict


if __name__ == '__main__':
    sample = '天氣真好呀優質浪漫na~我覺得其實OK辣哈如果有那麼好康的事情怎麼不跟窩講呢'

    print(sample)
    print(EDA(sample))
