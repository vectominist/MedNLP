'''
Data Augmentation
'''

import torch
import numpy as np


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
    
    n_sent = (input_dict['attention_mask'].sum(1) > 2).long().sum()
    swap_n_sent = np.random.randint(1, int(n_sent * max_swap_ratio))
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
    remove_n_sent = np.random.randint(1, int(n_sent * max_removal_ratio))
    indices = np.random.permutation(n_sent)[:-remove_n_sent]
    indices = np.sort(indices)
    output_dict = {}
    for key, val in input_dict.items():
        output_dict[key] = val[indices, :]
        if len(indices) < n_sent:
            output_dict[key] = torch.cat(
                [output_dict[key]] + [val[-1:, :]] * (n_sent - len(indices)), dim=0)
    return output_dict
