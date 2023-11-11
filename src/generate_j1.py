#!/usr/bin/env python
import argparse
import logging
import numpy as np
import pandas as pd
import time

from kaggler.data_io import save_data
from kaggler.preprocessing import LabelEncoder

from const import ID_COL, TARGET_COL


def generate_feature(train_file, test_file, train_feature_file,
                     test_feature_file, feature_header_file):
    logging.info('loading raw data')
    trn = pd.read_csv(train_file, index_col=ID_COL)
    tst = pd.read_csv(test_file, index_col=ID_COL)

    y = trn[TARGET_COL].values
    n_trn = trn.shape[0]

    trn.drop(TARGET_COL, axis=1, inplace=True)

    cat_cols = [x for x in trn.columns if trn[x].dtype == np.object]
    num_cols = [x for x in trn.columns if trn[x].dtype != np.object]

    logging.info(f'categorical: {len(cat_cols)}, numerical: {len(num_cols)}')

    df = pd.concat([trn, tst], axis=0)

    logging.info('label encoding categorical variables')
    lbe = LabelEncoder(min_obs=10)
    df[cat_cols] = lbe.fit_transform(df[cat_cols])
    df[num_cols] = df[num_cols].fillna(-1)

    with open(feature_header_file, 'w') as f:
        for i, col in enumerate(df.columns):
            f.write(f'{col}\n')

    logging.info('saving features')
    save_data(df.values[:n_trn, ], y, train_feature_file)
    save_data(df.values[n_trn:, ], None, test_feature_file)


if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s', level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--train-feature-file', required=True, dest='train_feature_file')
    parser.add_argument('--test-feature-file', required=True, dest='test_feature_file')
    parser.add_argument('--feature-header-file', required=True, dest='feature_header_file')

    args = parser.parse_args()

    start = time.time()
    generate_feature(args.train_file,
                     args.test_file,
                     args.train_feature_file,
                     args.test_feature_file,
                     args.feature_header_file)
    logging.info(f'finished ({time.time() - start:.2f} sec elasped)')
