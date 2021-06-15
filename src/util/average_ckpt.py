'''
    File      [ src/util/average_ckpt.py ]
    Author    [ Heng-Jui Chang (NTUEE) ]
    Synopsis  [ Average checkpoints ]
'''

import argparse
import os
import torch


def find_ckpts(path, files):
    ckpts = []
    for f in files:
        model_path = os.path.join(path, 'checkpoint-' + f, 'pytorch_model.bin')
        assert os.path.exists(model_path)
        ckpts.append(model_path)
    return ckpts


def average_ckpts(ckpts, out_file):
    # ref: ESPnet utils/average_checkpoints.py
    avg = None

    # sum
    for c in ckpts:
        ckpt = torch.load(c, map_location=torch.device('cpu'))
        states = ckpt
        if avg is None:
            avg = states
        else:
            for k in avg.keys():
                avg[k] += states[k]
    # average
    for k in avg.keys():
        if avg[k] is not None:
            if avg[k].is_floating_point():
                avg[k] /= len(ckpts)
            else:
                avg[k] //= len(ckpts)

    torch.save(avg, out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Average ckpts')
    parser.add_argument('--dir', type=str, help='Directory of ckpts.')
    parser.add_argument('--ckpts', type=str, nargs='+', help='ckpts.')
    parser.add_argument('--out', type=str, help='Output of the averaged ckpt.')
    args = parser.parse_args()

    ckpts = find_ckpts(args.dir, args.ckpts)
    average_ckpts(ckpts, args.out)
