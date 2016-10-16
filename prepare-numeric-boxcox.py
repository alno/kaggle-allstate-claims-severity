import numpy as np

from scipy.stats import skew, boxcox

from tqdm import tqdm
from util import Dataset

print "Loading data..."

train_num = Dataset.load_part('train', 'numeric')
test_num = Dataset.load_part('test', 'numeric')

train_num_enc = np.zeros(train_num.shape, dtype=np.float32)
test_num_enc = np.zeros(test_num.shape, dtype=np.float32)

with tqdm(total=train_num.shape[1], desc='  Transforming', unit='cols') as pbar:
    for col in xrange(train_num.shape[1]):
        values = np.hstack((train_num[:, col], test_num[:, col]))

        sk = skew(values)

        if sk > 0.25:
            values_enc, lam = boxcox(values+1)

            train_num_enc[:, col] = values_enc[:train_num.shape[0]]
            test_num_enc[:, col] = values_enc[train_num.shape[0]:]
        else:
            train_num_enc[:, col] = train_num[:, col]
            test_num_enc[:, col] = test_num[:, col]

        pbar.update(1)

print "Saving..."

Dataset.save_part_features('numeric_boxcox', Dataset.get_part_features('numeric'))
Dataset(numeric_boxcox=train_num_enc).save('train')
Dataset(numeric_boxcox=test_num_enc).save('test')

print "Done."
