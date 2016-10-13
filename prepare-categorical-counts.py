import pandas as pd

from tqdm import tqdm
from util import Dataset

print "Loading data..."

train_cat = Dataset.load_part('train', 'categorical')
test_cat = Dataset.load_part('test', 'categorical')

all_cat = pd.concat((train_cat, test_cat))

train_cat_counts = pd.DataFrame(0, columns=train_cat.columns, index=train_cat.index)
test_cat_counts = pd.DataFrame(0, columns=test_cat.columns, index=test_cat.index)

with tqdm(total=len(all_cat.columns), desc='  Counting', unit='cols') as pbar:
    for col in all_cat.columns:
        counts = all_cat[col].value_counts()

        train_cat_counts[col] = train_cat[col].map(counts)
        test_cat_counts[col] = test_cat[col].map(counts)

        pbar.update(1)

print "Saving..."

Dataset(categorical_counts=train_cat_counts).save('train')
Dataset(categorical_counts=test_cat_counts).save('test')

print "Done."
