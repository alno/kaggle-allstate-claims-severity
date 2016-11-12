import numpy as np
import scipy.sparse as sp

from tqdm import tqdm
from util import Dataset

print "Loading data..."

min_freq = 10

train_cat = Dataset.load_part('train', 'categorical')
test_cat = Dataset.load_part('test', 'categorical')

train_cat_enc = []
test_cat_enc = []

cats = Dataset.get_part_features('categorical')
features = []

with tqdm(total=len(cats), desc='  Encoding', unit='cols') as pbar:
    for col, cat in enumerate(cats):
        value_counts = dict(zip(*np.unique(train_cat[:, col], return_counts=True)))

        train_rares = np.zeros(train_cat.shape[0], dtype=np.uint8)
        test_rares = np.zeros(test_cat.shape[0], dtype=np.uint8)

        for val in value_counts:
            if value_counts[val] >= min_freq:
                features.append('%s_%s' % (cat, val))
                train_cat_enc.append(sp.csr_matrix((train_cat[:, col] == val).astype(np.uint8).reshape((train_cat.shape[0], 1))))
                test_cat_enc.append(sp.csr_matrix((test_cat[:, col] == val).astype(np.uint8).reshape((test_cat.shape[0], 1))))
            else:
                train_rares += (train_cat[:, col] == val).astype(np.uint8)
                test_rares += (test_cat[:, col] == val).astype(np.uint8)

        if train_rares.sum() > 0 and test_rares.sum() > 0:
            features.append('%s_rare' % cat)
            train_cat_enc.append(sp.csr_matrix(train_rares.reshape((train_cat.shape[0], 1))))
            test_cat_enc.append(sp.csr_matrix(test_rares.reshape((test_cat.shape[0], 1))))

        pbar.update(1)

print "Created %d dummy vars" % len(features)

print "Saving..."

train_cat_enc = sp.hstack(train_cat_enc, format='csr')
test_cat_enc = sp.hstack(test_cat_enc, format='csr')

Dataset.save_part_features('categorical_dummy', features)
Dataset(categorical_dummy=train_cat_enc).save('train')
Dataset(categorical_dummy=test_cat_enc).save('test')

print "Done."
