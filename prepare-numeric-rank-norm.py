import numpy as np

from scipy.stats.mstats import rankdata
from scipy.special import erfinv

from sklearn.preprocessing import scale, minmax_scale

from tqdm import tqdm
from util import Dataset

print "Loading data..."

lim = 0.999

train_num = Dataset.load_part('train', 'numeric')
test_num = Dataset.load_part('test', 'numeric')

train_num_enc = np.zeros(train_num.shape, dtype=np.float32)
test_num_enc = np.zeros(test_num.shape, dtype=np.float32)

with tqdm(total=train_num.shape[1], desc='  Transforming', unit='cols') as pbar:
    for col in xrange(train_num.shape[1]):
        values = np.hstack((train_num[:, col], test_num[:, col]))

        # Apply rank transformation
        values = rankdata(values).astype(np.float64)

        # Scale into range (-1, 1)
        values = minmax_scale(values, feature_range=(-lim, lim))

        # Make gaussian
        values = scale(erfinv(values))

        train_num_enc[:, col] = values[:train_num.shape[0]]
        test_num_enc[:, col] = values[train_num.shape[0]:]

        pbar.update(1)

print "Saving..."

Dataset.save_part_features('numeric_rank_norm', Dataset.get_part_features('numeric'))
Dataset(numeric_rank_norm=train_num_enc).save('train')
Dataset(numeric_rank_norm=test_num_enc).save('test')

print "Done."
