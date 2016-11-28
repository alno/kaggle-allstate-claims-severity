import pandas as pd
import numpy as np

from util import Dataset

for name in ['train', 'test']:
    print "Processing %s..." % name
    data = pd.read_csv('../input/lin_%s.csv' % name)

    # Save column names
    if name == 'train':
        num_columns = [c for c in data.columns if c.startswith('cont')]

        Dataset.save_part_features('numeric_lin', num_columns)

    Dataset(numeric_lin=data[num_columns].values.astype(np.float32)).save(name)

print "Done."
