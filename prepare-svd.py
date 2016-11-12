import numpy as np

from util import Dataset, vstack, hstack

from sklearn.preprocessing import scale
from sklearn.decomposition import TruncatedSVD

n_components = 500  # 500 components explain 99.8% of variance

print "Loading data..."

train_num = Dataset.load_part('train', 'numeric')
train_cat = Dataset.load_part('train', 'categorical_dummy')

test_num = Dataset.load_part('test', 'numeric')
test_cat = Dataset.load_part('test', 'categorical_dummy')

train_cnt = train_num.shape[0]

print "Combining data..."

all_data = hstack((scale(vstack((train_num, test_num)).astype(np.float64)).astype(np.float32), vstack((train_cat, test_cat))))

del train_num, train_cat, test_num, test_cat

print "Fitting svd..."

svd = TruncatedSVD(n_components)
res = svd.fit_transform(all_data)

print "Explained variance ratio: %.5f" % np.sum(svd.explained_variance_ratio_)

print "Saving..."

Dataset.save_part_features('svd', ['svd%d' % i for i in xrange(n_components)])
Dataset(svd=res[:train_cnt]).save('train')
Dataset(svd=res[train_cnt:]).save('test')

print "Done."
