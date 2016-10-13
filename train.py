import numpy as np
import pandas as pd
import xgboost as xgb

import argparse
import os
import datetime

from shutil import copy2

from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import KFold

from util import Dataset


class Xgb():

    def __init__(self, params, n_iter=400, transform_y=None):
        self.params = params
        self.n_iter = n_iter
        self.transform_y = transform_y

    def fit(self, X_train, y_train, X_eval=None, y_eval=None):
        if self.transform_y is not None:
            y_tr, y_inv = self.transform_y

            y_train = y_tr(y_train)
            y_eval = y_tr(y_eval)

            feval = lambda y_pred, y_true: ('mae', mean_absolute_error(y_inv(y_true.get_label()), y_inv(y_pred)))
        else:
            feval = None

        dtrain = xgb.DMatrix(X_train.values, label=y_train.values)
        deval = xgb.DMatrix(X_eval.values, label=y_eval.values)

        self.model = xgb.train(self.params, dtrain, self.n_iter, [(deval, 'eval'), (dtrain, 'train')], verbose_eval=10, feval=feval)

    def predict(self, X):
        pred = self.model.predict(xgb.DMatrix(X.values))

        if self.transform_y is not None:
            _, y_inv = self.transform_y

            pred = y_inv(pred)

        return pd.Series(pred, index=X.index)


parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('preset', type=str, help='model preset (features and hyperparams)')

args = parser.parse_args()

n_folds = 5

presets = {
    'xgb1': {
        'feature_parts': ['numeric', 'categorical_encoded'],
        'model': Xgb({
            'max_depth': 7,
            'eta': 0.1,
            'objective': 'reg:linear',
            'colsample_bytree': 0.5,
            'subsample': 0.95,
            'min_child_weight': 5,
            'eval_metric': 'mae',
            'silent': 1,
            'seed': 42,
            'nthread': 4
        }, n_iter=400, transform_y=(np.log, np.exp)),
    },
}

preset = presets[args.preset]

print "Loading train data..."
train = Dataset.load('train', preset['feature_parts'] + ['loss'])
train_x = pd.concat([train[part] for part in preset['feature_parts']], axis=1)
train_y = train['loss']
train_p = pd.Series(np.nan, index=train_x.index)
del train

print "Loading test data..."
test = Dataset.load('test', preset['feature_parts'])
test_x = pd.concat([test[part] for part in preset['feature_parts']], axis=1)
test_p = pd.Series(0.0, index=test_x.index)
del test

maes = []

print "Training..."
for fold, (fold_train_idx, fold_eval_idx) in enumerate(KFold(len(train_y), n_folds, shuffle=True, random_state=2016)):
    print
    print "  Fold %d..." % fold

    fold_train_x = train_x.iloc[fold_train_idx]
    fold_train_y = train_y.iloc[fold_train_idx]

    fold_eval_x = train_x.iloc[fold_eval_idx]
    fold_eval_y = train_y.iloc[fold_eval_idx]

    model = preset['model']
    model.fit(fold_train_x, fold_train_y, fold_eval_x, fold_eval_y)

    fold_eval_p = model.predict(fold_eval_x)
    fold_test_p = model.predict(test_x)

    train_p.loc[fold_eval_p.index] = fold_eval_p
    test_p += fold_test_p

    maes.append(mean_absolute_error(fold_eval_y, fold_eval_p))

## Analyzing predictions

test_p /= n_folds

mae_mean = np.mean(maes)
mae_std = np.std(maes)

print
print "CV MAE: %.5f +- %.5f" % (mae_mean, mae_std)

name = "%s-%s-%.5f" % (datetime.datetime.now().strftime('%Y%m%d-%H%M'), args.preset, mae_mean)

print "Saving..."

for part, pred in [('train', train_p), ('test', test_p)]:
    pred.rename('loss', inplace=True)
    pred.index.rename('id', inplace=True)
    pred.to_csv('preds/%s-%s.csv' % (name, part), header=True)

copy2(os.path.realpath(__file__), os.path.join("preds", "%s-code.py" % name))

print "Done."
