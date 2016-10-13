import pandas as pd

from util import Dataset

for name in ['train', 'test']:
    print "Processing %s..." % name
    data = pd.read_csv('../input/%s.csv.zip' % name, index_col='id')

    cat_columns = [c for c in data.columns if c.startswith('cat')]
    Dataset(categorical=data[cat_columns]).save(name)

    num_columns = [c for c in data.columns if c.startswith('cont')]
    Dataset(numeric=data[num_columns]).save(name)

    if 'loss' in data.columns:
        Dataset(loss=data['loss']).save(name)

print "Done."
