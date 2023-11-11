#!/usr/bin/env python
from sklearn.metrics import roc_auc_score as eval_func
import argparse
import numpy as np
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--target-file', '-t', required=True, dest='target_file')
    parser.add_argument('--predict-file', '-p', required=True, dest='predict_file')

    args = parser.parse_args()

    p = np.loadtxt(args.predict_file, delimiter=',')
    y = np.loadtxt(args.target_file, delimiter=',')

    model_name = os.path.splitext(os.path.splitext(os.path.basename(args.predict_file))[0])[0]
    print(f'{model_name}\t{eval_func(y, p):.6f}')
