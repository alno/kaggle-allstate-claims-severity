import numpy as np
import scipy.sparse as sp

from tqdm import tqdm
from util import Dataset

print "Loading data..."

num_features = Dataset.get_part_features('numeric')

train_num = Dataset.load_part('train', 'numeric')
test_num = Dataset.load_part('test', 'numeric')

train_n = train_num.shape[0]

features = []
train_res = []
test_res = []

with tqdm(total=train_num.shape[1], desc='  Transforming', unit='cols') as pbar:
    for col, col_name in enumerate(num_features):
        values = np.hstack((train_num[:, col], test_num[:, col]))

        if (values == 0.0).sum() > 20:
            features.append(col_name + '_zero')
            train_res.append((values[:train_n] == 0.0).astype(np.uint8).reshape((train_num.shape[0], 1)))
            test_res.append((values[train_n:] == 0.0).astype(np.uint8).reshape((test_num.shape[0], 1)))

        if (values == 1.0).sum() > 20:
            features.append(col_name + '_one')
            train_res.append((values[:train_n] == 1.0).astype(np.uint8).reshape((train_num.shape[0], 1)))
            test_res.append((values[train_n:] == 1.0).astype(np.uint8).reshape((test_num.shape[0], 1)))

        pbar.update(1)

print "Saving..."

Dataset.save_part_features('numeric_edges', features)
Dataset(numeric_edges=sp.csr_matrix(np.hstack(train_res))).save('train')
Dataset(numeric_edges=sp.csr_matrix(np.hstack(test_res))).save('test')

print "Done."
