import pandas as pd

from tqdm import tqdm
from util import Dataset

print "Loading data..."

train_cat = Dataset.load_part('train', 'categorical')
test_cat = Dataset.load_part('test', 'categorical')

train_cat_enc = pd.DataFrame(0, columns=train_cat.columns, index=train_cat.index)
test_cat_enc = pd.DataFrame(0, columns=test_cat.columns, index=test_cat.index)

with tqdm(total=len(train_cat.columns), desc='  Encoding', unit='cols') as pbar:
    for col in train_cat.columns:
        values = pd.concat((train_cat[col], test_cat[col]))
        values = sorted(values.unique(), key=lambda x: (len(x), x))

        encoding = dict(zip(values, range(len(values))))

        train_cat_enc[col] = train_cat[col].map(encoding)
        test_cat_enc[col] = test_cat[col].map(encoding)

        pbar.update(1)

print "Saving..."

Dataset(categorical_encoded=train_cat_enc).save('train')
Dataset(categorical_encoded=test_cat_enc).save('test')

print "Done."
