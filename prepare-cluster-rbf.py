import numpy as np

from util import Dataset, vstack, hstack

from sklearn.preprocessing import scale
from sklearn.cluster import MiniBatchKMeans

np.random.seed(1234)

n_clusters = 200
gamma = 1.0

print "Loading data..."

train_num = Dataset.load_part('train', 'numeric')
train_cat = Dataset.load_part('train', 'categorical_dummy')

test_num = Dataset.load_part('test', 'numeric')
test_cat = Dataset.load_part('test', 'categorical_dummy')

print "Combining data..."

all_data = hstack((scale(vstack((train_num, test_num)).astype(np.float64)).astype(np.float32), vstack((train_cat, test_cat))))

print "Finding clusters..."

kmeans = MiniBatchKMeans(n_clusters)
kmeans.fit(all_data)

print "transforming data..."

cluster_rbf = np.exp(- gamma * kmeans.transform(all_data))

print "Saving..."

Dataset.save_part_features('cluster_rbf', ['cluster_rbf_%d' % i for i in xrange(n_clusters)])
Dataset(cluster_rbf=cluster_rbf[:train_num.shape[0]]).save('train')
Dataset(cluster_rbf=cluster_rbf[train_num.shape[0]:]).save('test')

print "Done."
