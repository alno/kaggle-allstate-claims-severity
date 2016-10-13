import numpy as np
import pandas as pd
import xgboost as xgb

import argparse
import os
import datetime

from shutil import copy2

from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import KFold
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from util import Dataset, load_prediction


class Xgb(object):

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


class Sklearn(object):

    def __init__(self, model, transform_y=None):
        self.model = model
        self.transform_y = transform_y

    def fit(self, X_train, y_train, X_eval=None, y_eval=None):
        if self.transform_y is not None:
            y_tr, _ = self.transform_y
            y_train = y_tr(y_train)

        self.model.fit(X_train.values, y_train.values)

    def predict(self, X):
        pred = self.model.predict(X.values)

        if self.transform_y is not None:
            _, y_inv = self.transform_y
            pred = y_inv(pred)

        return pd.Series(pred, index=X.index)


def load_x(ds, preset):
    feature_parts = [Dataset.load_part(ds, part) for part in preset.get('features', [])]
    prediction_parts = [load_prediction(ds, p) for p in preset.get('predictions', [])]

    if 'prediction_transform' in preset:
        prediction_parts = map(preset['prediction_transform'], prediction_parts)

    return pd.concat(feature_parts + prediction_parts, axis=1)


## Main part


parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('preset', type=str, help='model preset (features and hyperparams)')

args = parser.parse_args()

n_folds = 5

presets = {
    'xgb1': {
        'features': ['numeric', 'categorical_encoded'],
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

    'et1': {
        'features': ['numeric', 'categorical_encoded'],
        'model': Sklearn(ExtraTreesRegressor(50, n_jobs=-1), transform_y=(np.log, np.exp)),
    },

    'rf1': {
        'features': ['numeric', 'categorical_encoded'],
        'model': Sklearn(RandomForestRegressor(50, n_jobs=-1), transform_y=(np.log, np.exp)),
    },

    'lr1': {
        'features': ['numeric', 'categorical_dummy'],
        'model': Sklearn(Pipeline([('sc', StandardScaler()), ('lr', Ridge(1e-3))]), transform_y=(np.log, np.exp)),
    },

    'knn1': {
        'features': ['numeric', 'categorical_encoded'],
        'model': Sklearn(Pipeline([('sc', StandardScaler()), ('knn', KNeighborsRegressor(5))]), transform_y=(np.log, np.exp)),
    },

    'l2_lr': {
        'predictions': [
            '20161013-1512-xgb1-1146.11469',
            '20161013-1606-et1-1227.13876',
            '20161013-1532-lr1-1290.94745',
            '20161013-1546-lr1-1250.76315',
        ],
        'prediction_transform': np.log,
        'model': Sklearn(Ridge(), transform_y=(np.log, np.exp)),
    },
}

preset = presets[args.preset]

print "Loading train data..."
train_x = load_x('train', preset)
train_y = Dataset.load_part('train', 'loss')
train_p = pd.Series(np.nan, index=train_x.index)

print "Loading test data..."
test_x = load_x('test', preset)
test_p = pd.Series(0.0, index=test_x.index)

maes = []

print "Training..."
for fold, (fold_train_idx, fold_eval_idx) in enumerate(KFold(len(train_y), n_folds, shuffle=True, random_state=2016)):
    print
    print "  Fold %d..." % fold

    fold_train_x = train_x.iloc[fold_train_idx]
    fold_train_y = train_y.iloc[fold_train_idx]

    fold_eval_x = train_x.iloc[fold_eval_idx]
    fold_eval_y = train_y.iloc[fold_eval_idx]

    # Fit model
    model = preset['model']
    model.fit(fold_train_x, fold_train_y, fold_eval_x, fold_eval_y)

    del fold_train_x, fold_train_y

    # Predict on eval
    fold_eval_p = model.predict(fold_eval_x)
    fold_mae = mean_absolute_error(fold_eval_y, fold_eval_p)

    train_p.loc[fold_eval_p.index] = fold_eval_p

    del fold_eval_x, fold_eval_y

    maes.append(fold_mae)

    print "  MAE: %.5f" % fold_mae

    # Predict on test
    test_p += model.predict(test_x)

## Analyzing predictions

test_p /= n_folds

mae_mean = np.mean(maes)
mae_std = np.std(maes)

print
print "CV MAE: %.5f +- %.5f" % (mae_mean, mae_std)

name = "%s-%s-%.5f" % (datetime.datetime.now().strftime('%Y%m%d-%H%M'), args.preset, mae_mean)

print
print "Saving predictions... (%s)" % name

for part, pred in [('train', train_p), ('test', test_p)]:
    pred.rename('loss', inplace=True)
    pred.index.rename('id', inplace=True)
    pred.to_csv('preds/%s-%s.csv' % (name, part), header=True)

copy2(os.path.realpath(__file__), os.path.join("preds", "%s-code.py" % name))

print "Done."
