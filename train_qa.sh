#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python src/train_qa.py --config config/qa.yaml --mode train
