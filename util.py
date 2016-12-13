import numpy as np
import pandas as pd

import scipy.sparse as sp

import cPickle as pickle
import os

n_folds = 5

cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')


def hstack(x):
    if any(sp.issparse(p) for p in x):
        return sp.hstack(x, format='csr')
    else:
        return np.hstack(x)


def vstack(x):
    if any(sp.issparse(p) for p in x):
        return sp.vstack(x, format='csr')
    else:
        return np.vstack(x)


def save_pickle(filename, data):
    with open(filename, 'w') as f:
        pickle.dump(data, f)


def load_pickle(filename):
    with open(filename) as f:
        return pickle.load(f)


def save_csr(filename, array):
    np.savez_compressed(filename, data=array.data, indices=array.indices, indptr=array.indptr, shape=array.shape)


def load_csr(filename):
    loader = np.load(filename)

    return sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])


def load_prediction(split, name, mode='fulltrain'):
    if type(name) is tuple:
        pred = load_prediction(split, name[0], mode)
        opts = name[1]

        if 'power' in opts:
            pred = pred ** opts['power']

        return pred
    if type(name) is list:
        preds = [load_prediction(split, n, mode) for n in name]

        return sum(preds) / len(preds)
    else:
        if split == 'test' and os.path.exists('preds/%s-%s.csv' % (name, 'test-%s' % mode)):
            return pd.read_csv('preds/%s-%s.csv' % (name, 'test-%s' % mode), index_col='id').iloc[:, 0]

        return pd.read_csv('preds/%s-%s.csv' % (name, split), index_col='id').iloc[:, 0]


class Dataset(object):

    part_types = {
        'id': 'd1',
        'loss': 'd1',
        'numeric': 'd2',
        'numeric_lin': 'd2',
        'numeric_scaled': 'd2',
        'numeric_boxcox': 'd2',
        'numeric_rank_norm': 'd2',
        'numeric_combinations': 'd2',
        'numeric_edges': 's2',
        'numeric_unskew': 'd2',
        'categorical': 'd2',
        'categorical_counts': 'd2',
        'categorical_encoded': 'd2',
        'categorical_dummy': 's2',
        'svd': 'd2',
        'manual': 'd2',
        'cluster_rbf_25': 'd2',
        'cluster_rbf_50': 'd2',
        'cluster_rbf_75': 'd2',
        'cluster_rbf_100': 'd2',
        'cluster_rbf_200': 'd2',
    }

    parts = part_types.keys()

    @classmethod
    def save_part_features(cls, part_name, features):
        save_pickle('%s/%s-features.pickle' % (cache_dir, part_name), features)

    @classmethod
    def get_part_features(cls, part_name):
        return load_pickle('%s/%s-features.pickle' % (cache_dir, part_name))

    @classmethod
    def load(cls, name, parts):
        return cls(**{part_name: cls.load_part(name, part_name) for part_name in parts})

    @classmethod
    def load_part(cls, name, part_name):
        if cls.part_types[part_name][0] == 's':
            return load_csr('%s/%s-%s.npz' % (cache_dir, part_name, name))
        else:
            return np.load('%s/%s-%s.npy' % (cache_dir, part_name, name))

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
            self.save_part(part_name, name)

    def save_part(self, part_name, name):
        if self.part_types[part_name][0] == 's':
            save_csr('%s/%s-%s.npz' % (cache_dir, part_name, name), self.parts[part_name])
        else:
            np.save('%s/%s-%s.npy' % (cache_dir, part_name, name), self.parts[part_name])

    def slice(self, index):
        new_parts = {}

        for part_name in self.parts:
            new_parts[part_name] = self.parts[part_name][index]

        return Dataset(**new_parts)
