'''
    File      [ src/util/utils.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ Some utilities ]
'''

import torch


def count_parameters(model):
    ''' Count total trainable parameters in a nn.Module '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
