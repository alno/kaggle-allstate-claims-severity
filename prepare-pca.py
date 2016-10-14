import pandas as pd
import numpy as np

from util import Dataset

from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

parts = ['numeric', 'categorical_dummy']
n_components = 725  # 725 components explain 99% of variance

print "Loading data..."

train = pd.concat([Dataset.load_part('train', p) for p in parts], axis=1)
test = pd.concat([Dataset.load_part('test', p) for p in parts], axis=1)

print "Fitting pca..."

pca = PCA(n_components, copy=False)
res = pca.fit_transform(np.vstack((scale(train.values), test.values)))

columns = ['pca%d' % i for i in xrange(n_components)]

Dataset(pca=pd.DataFrame(res[:len(train)], columns=columns, index=train.index)).save('train')
Dataset(pca=pd.DataFrame(res[len(train):], columns=columns, index=test.index)).save('test')
