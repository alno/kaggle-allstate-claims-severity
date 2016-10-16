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
from sklearn.svm import LinearSVR

from keras.models import Sequential
from keras.layers import Dense, Dropout, Lambda
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Nadam, SGD, Adam
from keras.regularizers import l1l2
from keras import backend as K
from keras import initializations

from scipy.stats import boxcox

from util import Dataset, load_prediction


class Xgb(object):

    default_params = {
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'silent': 1,
        'seed': 42,
        'nthread': -1,
    }

    def __init__(self, params, n_iter=400, transform_y=None):
        self.params = self.default_params.copy()

        for k in params:
            self.params[k] = params[k]

        self.n_iter = n_iter
        self.transform_y = transform_y

    def fit(self, X_train, y_train, X_eval=None, y_eval=None):
        y_train = y_train.values
        y_eval = y_eval.values

        if self.transform_y is not None:
            y_tr, y_inv = self.transform_y

            y_train = y_tr(y_train)
            y_eval = y_tr(y_eval)

            feval = lambda y_pred, y_true: ('mae', mean_absolute_error(y_inv(y_true.get_label()), y_inv(y_pred)))
        else:
            feval = None

        dtrain = xgb.DMatrix(X_train.values, label=y_train)
        deval = xgb.DMatrix(X_eval.values, label=y_eval)

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
        y_train = y_train.values

        if self.transform_y is not None:
            y_tr, _ = self.transform_y
            y_train = y_tr(y_train)

        self.model.fit(X_train.values, y_train)

    def predict(self, X):
        pred = self.model.predict(X.values)

        if self.transform_y is not None:
            _, y_inv = self.transform_y
            pred = y_inv(pred)

        return pd.Series(pred, index=X.index)


class Keras(object):

    def __init__(self, params, transform_y=None):
        self.params = params
        self.transform_y = transform_y

    def fit(self, X_train, y_train, X_eval=None, y_eval=None):
        y_train = y_train.values
        y_eval = y_eval.values

        if self.transform_y is not None:
            y_tr, y_inv = self.transform_y

            y_train = y_tr(y_train)
            y_eval = y_tr(y_eval)

        reg = l1l2(self.params['l1'], self.params['l2'])

        self.scaler = StandardScaler()

        X_train = self.scaler.fit_transform(X_train.values)
        X_eval = self.scaler.transform(X_eval.values)

        self.model = Sequential()

        for i, layer_size in enumerate(self.params['layers']):
            if i == 0:
                self.model.add(Dense(layer_size, init='he_normal', W_regularizer=reg, input_shape=(X_train.shape[1],)))
            else:
                self.model.add(Dense(layer_size, init='he_normal', W_regularizer=reg))
            self.model.add(PReLU())

            if 'dropout' in self.params:
                self.model.add(Dropout(self.params['dropout']))

        self.model.add(Dense(1, input_shape=(X_train.shape[1],)))

        self.model.compile(optimizer=Adam(lr=self.params['lr']), loss='mae')

        self.model.fit(X_train, y_train, validation_data=(X_eval, y_eval), batch_size=self.params['batch_size'], nb_epoch=self.params['n_epoch'], verbose=1, shuffle=True)

    def predict(self, X):
        pred = self.model.predict(self.scaler.transform(X.values)).reshape((X.shape[0],))

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


norm_y_lambda = 0.7


def norm_y(y):
    return boxcox(np.log1p(y), lmbda=norm_y_lambda)


def norm_y_inv(y_bc):
    return np.expm1((y_bc * norm_y_lambda + 1)**(1/norm_y_lambda))


## Main part


parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('preset', type=str, help='model preset (features and hyperparams)')

args = parser.parse_args()

n_folds = 5

l1_predictions = [
    '20161013-1512-xgb1-1146.11469',
    '20161013-1606-et1-1227.13876',
    #'20161013-1546-lr1-1250.76315',
    '20161013-2256-lr2-1250.56353',
    '20161013-2323-xgb2-1147.11866',
    '20161014-1330-xgb3-1143.31331',
    #'20161015-0118-nn1-1179.19525',
    '20161016-1716-nn1-1172.44150',
]

l2_predictions = [
    '20161015-0118-l2_lr-1135.83902',
    '20161015-0120-l2_nn-1133.22684',
]

presets = {
    'xgb1': {
        'features': ['numeric', 'categorical_encoded'],
        'model': Xgb({
            'max_depth': 7,
            'eta': 0.1,
            'colsample_bytree': 0.5,
            'subsample': 0.95,
            'min_child_weight': 5,
        }, n_iter=400, transform_y=(norm_y, norm_y_inv)),
    },

    'xgb2': {
        'features': ['numeric', 'categorical_counts'],
        'model': Xgb({
            'max_depth': 7,
            'eta': 0.1,
            'colsample_bytree': 0.5,
            'subsample': 0.95,
            'min_child_weight': 5,
        }, n_iter=400, transform_y=(norm_y, norm_y_inv)),
    },

    'xgb3': {
        'features': ['numeric', 'categorical_encoded'],
        'model': Xgb({
            'max_depth': 7,
            'eta': 0.06,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'min_child_weight': 4,
        }, n_iter=550, transform_y=(norm_y, norm_y_inv)),
    },

    'nn1': {
        'features': ['numeric', 'categorical_encoded'],
        'n_bags': 2,
        'model': Keras({'l1': 1e-3, 'l2': 1e-3, 'lr': 1e-3, 'n_epoch': 100, 'batch_size': 48, 'layers': [400, 200], 'dropout': 0.2}),
    },

    'et1': {
        'features': ['numeric', 'categorical_encoded'],
        'model': Sklearn(ExtraTreesRegressor(50, max_depth=10, n_jobs=-1), transform_y=(np.log, np.exp)),
    },

    'rf1': {
        'features': ['numeric', 'categorical_encoded'],
        'model': Sklearn(RandomForestRegressor(50, max_depth=10, max_features=0.2, min_samples_leaf=2, n_jobs=-1), transform_y=(np.log, np.exp)),
    },

    'lr1': {
        'features': ['numeric', 'categorical_dummy'],
        'model': Sklearn(Pipeline([('sc', StandardScaler()), ('lr', Ridge(1e-3))]), transform_y=(np.log, np.exp)),
    },

    'lr2': {
        'features': ['numeric', 'categorical_encoded'],
        'model': Sklearn(Pipeline([('sc', StandardScaler()), ('lr', Ridge(1e-3))]), transform_y=(np.log, np.exp)),
    },

    'lr2': {
        'features': ['pca'],
        'model': Sklearn(Ridge(1e-3), transform_y=(np.log, np.exp)),
    },

    'knn1': {
        'features': ['pca'],
        'model': Sklearn(Pipeline([('sc', StandardScaler()), ('knn', KNeighborsRegressor(5))]), transform_y=(np.log, np.exp)),
    },

    'l2_nn': {
        'predictions': l1_predictions,
        'n_bags': 5,
        'model': Keras({'l1': 1e-3, 'l2': 1e-3, 'lr': 1e-2, 'n_epoch': 10, 'batch_size': 48, 'layers': [10]}),
    },

    'l2_lr': {
        'predictions': l1_predictions,
        'prediction_transform': np.log,
        'model': Sklearn(Ridge(), transform_y=(np.log, np.exp)),
    },

    'l2_xgb': {
        'features': ['categorical_encoded'],
        'predictions': l1_predictions,
        'prediction_transform': np.log,
        'model': Xgb({
            'max_depth': 4,
            'eta': 0.06,
            'colsample_bytree': 0.7,
            'subsample': 0.95,
            'min_child_weight': 10,
        }, n_iter=550, transform_y=(norm_y, norm_y_inv)),
    },

    'l3_nn': {
        'predictions': l2_predictions,
        'model': Keras({'l1': 1e-3, 'l2': 1e-3, 'lr': 1e-2, 'n_epoch': 10, 'batch_size': 48, 'layers': [10]}),
    },
}

preset = presets[args.preset]

n_bags = preset.get('n_bags', 1)

print "Loading train data..."
train_x = load_x('train', preset)
train_y = Dataset.load_part('train', 'loss')
train_p = pd.Series(0.0, index=train_x.index)

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

    for _ in xrange(n_bags):
        # Fit model
        model = preset['model']
        model.fit(fold_train_x, fold_train_y, fold_eval_x, fold_eval_y)

        # Predict on eval
        train_p.loc[fold_eval_x.index] += model.predict(fold_eval_x)

        # Predict on test
        test_p += model.predict(test_x)

    # Normalize train predictions
    train_p.loc[fold_eval_x.index] /= n_bags

    # Calculate err
    maes.append(mean_absolute_error(fold_eval_y, train_p.loc[fold_eval_x.index]))

    print "  MAE: %.5f" % maes[-1]

    # Free mem
    del fold_train_x, fold_train_y, fold_eval_x, fold_eval_y


## Analyzing predictions

test_p /= n_folds * n_bags

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
