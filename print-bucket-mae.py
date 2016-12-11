import pandas as pd
import sys

from sklearn.metrics import mean_absolute_error

from util import Dataset, load_prediction


df = pd.DataFrame({'loss': Dataset.load_part('train', 'loss')}, index=Dataset.load_part('train', 'id'))

edges = df['loss'].quantile([0.2, 0.4, 0.6, 0.8]).values

df['bucket'] = len(edges)
for i in reversed(xrange(len(edges))):
    df.loc[df['loss'] <= edges[i], 'bucket'] = i


pred = load_prediction('train', sys.argv[1])

errs = (pd.Series(pred, index=df.index) - df['loss']).abs()

print errs.groupby(df['bucket']).mean()
