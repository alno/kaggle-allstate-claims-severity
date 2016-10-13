import pandas as pd

from tqdm import tqdm
from util import Dataset

print "Loading data..."

train_cat = Dataset.load_part('train', 'categorical')
test_cat = Dataset.load_part('test', 'categorical')

train_cat_enc = pd.DataFrame(index=train_cat.index)
test_cat_enc = pd.DataFrame(index=test_cat.index)

with tqdm(total=len(train_cat.columns), desc='  Encoding', unit='cols') as pbar:
    for col in train_cat.columns:
        value_counts = train_cat[col].value_counts().to_dict()

        for val in value_counts:
            if value_counts[val] > 5:
                train_cat_enc['%s_%s' % (col, val)] = (train_cat[col] == val).astype(int)
                test_cat_enc['%s_%s' % (col, val)] = (test_cat[col] == val).astype(int)

        pbar.update(1)

print "Saving..."

Dataset(categorical_dummy=train_cat_enc).save('train')
Dataset(categorical_dummy=test_cat_enc).save('test')

print "Done."
