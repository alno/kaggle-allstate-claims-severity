import numpy as np
import pandas as pd

import pickle
import os

n_folds = 5

cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')


def hstack(x):
    if any(csc.issparse(p) for p in x):
        return csc.hstack(x, format='csr')
    else:
        return np.hstack(x)


def vstack(x):
    if any(csc.issparse(p) for p in x):
        return csc.vstack(x, format='csr')
    else:
        return np.vstack(x)


def save_pickle(filename, data):
    with open(filename, 'w') as f:
        pickle.dump(data, f)


def load_pickle(filename):
    with open(filename) as f:
        return pickle.load(f)


def load_prediction(split, name):
    return pd.read_csv('preds/%s-%s.csv' % (name, split), index_col='id').iloc[:, 0]


class Dataset(object):

    parts = ['loss', 'numeric', 'numeric_boxcox', 'categorical', 'categorical_counts', 'categorical_encoded', 'categorical_dummy']

    @classmethod
    def get_part_features(cls, part_name):
        return load_pickle('%s/%s-features.pickle' % (cache_dir, part_name))

    @classmethod
    def load(cls, name, parts):
        return cls(**{part_name: cls.load_part(name, part_name) for part_name in parts})

    @classmethod
    def load_part(cls, name, part_name):
        return pd.read_pickle('%s/%s-%s.pickle' % (cache_dir, part_name, name))

    @classmethod
    def concat(cls, datasets):
        datasets = list(datasets)

        if len(datasets) == 0:
            raise ValueError("Empty concat list")

        if len(datasets) == 1:
            return datasets[0]

        new_parts = {}

        for part_name in datasets[0].parts:
            new_parts[part_name] = pd.concat(part_name, [ds[part_name] for ds in datasets])

        return cls(**new_parts)

    def __init__(self, **parts):
        self.parts = parts

    def __getitem__(self, key):
        return self.parts[key]

    def save(self, name):
        for part_name in self.parts:
            self.parts[part_name].to_pickle('%s/%s-%s.pickle' % (cache_dir, part_name, name))

    def slice(self, index):
        new_parts = {}

        for part_name in self.parts:
            new_parts[part_name] = self.parts[part_name].iloc[index]

        return Dataset(**new_parts)
