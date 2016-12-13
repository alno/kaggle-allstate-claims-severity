import pandas as pd
import numpy as np

import sys

from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import KFold
from statsmodels.regression.quantile_regression import QuantReg

from util import Dataset

pred_name = sys.argv[1]

n_folds = 8

train_y = Dataset.load_part('train', 'loss')
train_x = pd.read_csv('preds/%s-train.csv' % pred_name)['loss'].values

orig_maes = []
corr_maes = []

for fold, (fold_train_idx, fold_eval_idx) in enumerate(KFold(len(train_y), n_folds, shuffle=True, random_state=2016)):
    fold_train_x = train_x[fold_train_idx]
    fold_train_y = train_y[fold_train_idx]

    fold_eval_x = train_x[fold_eval_idx]
    fold_eval_y = train_y[fold_eval_idx]

    model = QuantReg(fold_train_y, fold_train_x).fit(q=0.5)

    fold_eval_p = model.predict(fold_eval_x)

    orig_maes.append(mean_absolute_error(fold_eval_y, fold_eval_x))
    corr_maes.append(mean_absolute_error(fold_eval_y, fold_eval_p))

    print "Fold %d, orig MAE = %.5f, corr MAE = %.5f" % (fold, orig_maes[-1], corr_maes[-1])

print
print "Avg orig MAE = %.5f" % np.mean(orig_maes)
print "Avg corr MAE = %.5f" % np.mean(corr_maes)

print "Done."
