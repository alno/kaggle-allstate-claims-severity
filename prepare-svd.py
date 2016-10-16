import numpy as np

from util import Dataset, vstack, hstack

from sklearn.preprocessing import scale
from sklearn.decomposition import TruncatedSVD

n_components = 725  # 725 components explain 99% of variance

print "Loading data..."

train_num = Dataset.load_part('train', 'numeric')
train_cat = Dataset.load_part('train', 'categorical_dummy')

test_num = Dataset.load_part('test', 'numeric')
test_cat = Dataset.load_part('test', 'categorical_dummy')

print "Combining data..."

all_data = hstack((scale(vstack((train_num, test_num)).astype(np.float64)).astype(np.float32), vstack((train_cat, test_cat))))

print "Fitting svd..."

svd = TruncatedSVD(n_components)
res = svd.fit_transform(all_data)

print "Saving..."

Dataset.save_part_features('svd', ['svd%d' % i for i in xrange(n_components)])
Dataset(svd=res[:train_num.shape[0]]).save('train')
Dataset(svd=res[train_num.shape[0]:]).save('test')

print "Done."
