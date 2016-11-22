import numpy as np

from util import Dataset, vstack, hstack

from sklearn.preprocessing import scale
from sklearn.cluster import MiniBatchKMeans

np.random.seed(1234)

gamma = 1.0

print "Loading data..."

train_num = Dataset.load_part('train', 'numeric')
train_cat = Dataset.load_part('train', 'categorical_dummy')

test_num = Dataset.load_part('test', 'numeric')
test_cat = Dataset.load_part('test', 'categorical_dummy')

print "Combining data..."

all_data = hstack((scale(vstack((train_num, test_num)).astype(np.float64)).astype(np.float32), vstack((train_cat, test_cat))))

for n_clusters in [50, 100, 200]:
    part_name = 'cluster_rbf_%d' % n_clusters

    print "Finding %d clusters..." % n_clusters

    kmeans = MiniBatchKMeans(n_clusters)
    kmeans.fit(all_data)

    print "transforming data..."

    cluster_rbf = np.exp(- gamma * kmeans.transform(all_data))

    print "Saving..."

    Dataset.save_part_features(part_name, ['cluster_rbf_%d_%d' % (n_clusters, i) for i in xrange(n_clusters)])
    Dataset(**{part_name: cluster_rbf[:train_num.shape[0]]}).save('train')
    Dataset(**{part_name: cluster_rbf[train_num.shape[0]:]}).save('test')

print "Done."
