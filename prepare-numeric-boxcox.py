import pandas as pd

from scipy.stats import skew, boxcox

from tqdm import tqdm
from util import Dataset

print "Loading data..."

train_num = Dataset.load_part('train', 'numeric')
test_num = Dataset.load_part('test', 'numeric')

train_num_enc = pd.DataFrame(0, columns=train_num.columns, index=train_num.index)
test_num_enc = pd.DataFrame(0, columns=test_num.columns, index=test_num.index)

with tqdm(total=len(train_num.columns), desc='  Transforming', unit='cols') as pbar:
    for col in train_num.columns:
        values = pd.concat((train_num[col], test_num[col]))

        sk = skew(values.dropna())

        if sk > 0.25:
            values_enc, lam = boxcox(values.values+1)

            train_num_enc[col] = values_enc[:train_num.shape[0]]
            test_num_enc[col] = values_enc[train_num.shape[0]:]
        else:
            train_num_enc[col] = train_num[col]
            test_num_enc[col] = test_num[col]

        pbar.update(1)

print "Saving..."

Dataset(numeric_boxcox=train_num_enc).save('train')
Dataset(numeric_boxcox=test_num_enc).save('test')

print "Done."
