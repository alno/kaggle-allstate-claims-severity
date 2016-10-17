import numpy as np
import pandas as pd
import xgboost as xgb
import scipy.sparse as sp

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
from sklearn.neighbors import LSHForest

from keras.models import Sequential
from keras.layers import Dense, Dropout, Lambda
from keras.layers.advanced_activations import PReLU
from keras.regularizers import l1l2
from keras import backend as K
from keras import initializations

from scipy.stats import boxcox

from tqdm import tqdm
from util import Dataset, load_prediction, hstack


def batch_generator(X, y=None, batch_size=128, shuffle=False):
    index = np.arange(X.shape[0])

    while True:
        if shuffle:
            np.random.shuffle(index)

        batch_start = 0
        while batch_start < X.shape[0]:
            batch_index = index[batch_start:batch_start + batch_size]
            batch_start += batch_size

            X_batch = X[batch_index, :]

            if sp.issparse(X_batch):
                X_batch = X_batch.toarray()

            if y is None:
                yield X_batch
            else:
                yield X_batch, y[batch_index]


class CategoricalMeanEncoded(object):

    requirements = ['categorical']

    def __init__(self, C=100, loo=False, noisy=True, random_state=11):
        self.random_state = np.random.RandomState(random_state)
        self.C = C
        self.loo = loo
        self.noisy = noisy

    def fit_transform(self, ds):
        train_cat = ds['categorical']
        train_target = pd.Series(np.log(ds['loss']))
        train_res = np.zeros(train_cat.shape, dtype=np.float32)

        self.global_target_mean = train_target.mean()
        self.global_target_std = train_target.std()

        self.target_sums = {}
        self.target_cnts = {}

        for col in xrange(train_cat.shape[1]):
            train_series = pd.Series(train_cat[:, col])

            self.target_sums[col] = train_target.groupby(train_series).sum()
            self.target_cnts[col] = train_target.groupby(train_series).count()

            if self.noisy:
                train_res_reg = self.random_state.normal(
                    loc=self.global_target_mean * self.C,
                    scale=self.global_target_std * np.sqrt(self.C),
                    size=len(train_series)
                )
            else:
                train_res_reg = self.global_target_mean * self.C

            train_res_num = train_series.map(self.target_sums[col]) + train_res_reg
            train_res_den = train_series.map(self.target_cnts[col]) + self.C

            if self.loo:  # Leave-one-out mode, exclude current observation
                train_res_num -= train_target
                train_res_den -= 1

            train_res[:, col] = np.exp(train_res_num / train_res_den).values

        return train_res

    def transform(self, ds):
        test_cat = ds['categorical']
        test_res = np.zeros(test_cat.shape, dtype=np.float32)

        for col in xrange(test_res.shape[1]):
            test_series = pd.Series(test_res[:, col])

            test_res_num = test_series.map(self.target_sums[col]).fillna(0.0) + self.global_target_mean * self.C
            test_res_den = test_series.map(self.target_cnts[col]).fillna(0.0) + self.C

            test_res[:, col] = np.exp(test_res_num / test_res_den).values

        return test_res


class Xgb(object):

    default_params = {
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'silent': 1,
        'nthread': -1,
    }

    def __init__(self, params, n_iter=400, transform_y=None):
        self.params = self.default_params.copy()

        for k in params:
            self.params[k] = params[k]

        self.n_iter = n_iter
        self.transform_y = transform_y

    def fit(self, X_train, y_train, X_eval=None, y_eval=None, seed=42):
        if self.transform_y is not None:
            y_tr, y_inv = self.transform_y

            y_train = y_tr(y_train)
            y_eval = y_tr(y_eval)

            feval = lambda y_pred, y_true: ('mae', mean_absolute_error(y_inv(y_true.get_label()), y_inv(y_pred)))
        else:
            feval = None

        dtrain = xgb.DMatrix(X_train, label=y_train)
        deval = xgb.DMatrix(X_eval, label=y_eval)

        params = self.params.copy()
        params['seed'] = seed

        self.model = xgb.train(params, dtrain, self.n_iter, [(deval, 'eval'), (dtrain, 'train')], verbose_eval=10, feval=feval)

    def predict(self, X):
        pred = self.model.predict(xgb.DMatrix(X))

        if self.transform_y is not None:
            _, y_inv = self.transform_y

            pred = y_inv(pred)

        return pred


class Sklearn(object):

    def __init__(self, model, transform_y=None):
        self.model = model
        self.transform_y = transform_y

    def fit(self, X_train, y_train, X_eval=None, y_eval=None, seed=42):
        if self.transform_y is not None:
            y_tr, _ = self.transform_y
            y_train = y_tr(y_train)

        self.model.fit(X_train, y_train)

    def predict(self, X):
        pred = self.model.predict(X)

        if self.transform_y is not None:
            _, y_inv = self.transform_y
            pred = y_inv(pred)

        return pred


class LshForest(object):

    def __init__(self, transform_y=None):
        self.transform_y = transform_y

    def fit(self, X_train, y_train, X_eval=None, y_eval=None, seed=42):
        if self.transform_y is not None:
            y_tr, _ = self.transform_y
            y_train = y_tr(y_train)

        self.scaler = StandardScaler()
        self.forest = LSHForest()
        self.y_train = y_train

        self.forest.fit(self.scaler.fit_transform(X_train))

    def predict(self, X):
        neighbors, _ = self.forest.kneighbors(self.scaler.transform(X))
        targets = self.y_train[neighbors]

        pred = targets.mean(axis=1)

        if self.transform_y is not None:
            _, y_inv = self.transform_y
            pred = y_inv(pred)

        return pred


class Keras(object):

    def __init__(self, params, transform_y=None, scale=True):
        self.params = params
        self.transform_y = transform_y
        self.scale = scale

    def fit(self, X_train, y_train, X_eval=None, y_eval=None, seed=42):
        if self.transform_y is not None:
            y_tr, y_inv = self.transform_y

            y_train = y_tr(y_train)
            y_eval = y_tr(y_eval)

        reg = l1l2(self.params['l1'], self.params['l2'])

        if self.scale:
            self.scaler = StandardScaler(with_mean=False)

            X_train = self.scaler.fit_transform(X_train)
            X_eval = self.scaler.transform(X_eval)

        self.model = Sequential()

        for i, layer_size in enumerate(self.params['layers']):
            if i == 0:
                self.model.add(Dense(layer_size, init='he_normal', W_regularizer=reg, input_shape=(X_train.shape[1],)))
            else:
                self.model.add(Dense(layer_size, init='he_normal', W_regularizer=reg))

            self.model.add(PReLU())

            if 'dropouts' in self.params:
                self.model.add(Dropout(self.params['dropouts'][i]))

        self.model.add(Dense(1, init='he_normal'))

        self.model.compile(optimizer='adadelta', loss='mae')

        self.model.fit_generator(
            generator=batch_generator(X_train, y_train, self.params['batch_size'], True), samples_per_epoch=X_train.shape[0],
            validation_data=batch_generator(X_eval, y_eval, 800), nb_val_samples=X_eval.shape[0],
            nb_epoch=self.params['n_epoch'], verbose=1)

    def predict(self, X):
        if self.scale:
            X = self.scaler.transform(X)

        pred = self.model.predict_generator(batch_generator(X, batch_size=800), val_samples=X.shape[0]).reshape((X.shape[0],))

        if self.transform_y is not None:
            _, y_inv = self.transform_y
            pred = y_inv(pred)

        return pred


def load_x(ds, preset):
    feature_parts = [Dataset.load_part(ds, part) for part in preset.get('features', [])]
    prediction_parts = [load_prediction(ds, p) for p in preset.get('predictions', [])]
    prediction_parts = [p.reshape((p.shape[0], 1)) for p in prediction_parts]

    if 'prediction_transform' in preset:
        prediction_parts = map(preset['prediction_transform'], prediction_parts)

    return hstack(feature_parts + prediction_parts)


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
    #'20161017-1704-rf1-1202.08857',
    #'20161013-1546-lr1-1250.76315',
    '20161013-2256-lr2-1250.56353',
    #'20161013-2323-xgb2-1147.11866',
    '20161014-1330-xgb3-1143.31331',
    '20161017-0645-xgb3-1137.53294',
    #'20161015-0118-nn1-1179.19525',
    #'20161016-1716-nn1-1172.44150',
    '20161016-2155-nn2-1142.81539',
    '20161017-0252-nn2-1138.89229',

    '20161017-1350-nn4-1165.61897',
]

l2_predictions = [
    '20161015-0118-l2_lr-1135.83902',
    '20161015-0120-l2_nn-1133.22684',
    '20161017-0116-l2_nn-1129.48210',
]

presets = {
    'xgb-tst': {
        'features': ['numeric'],
        'model': Xgb({}, n_iter=10),
    },

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
        'n_bags': 3,
        'model': Xgb({
            'max_depth': 7,
            'eta': 0.03,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'min_child_weight': 4,
        }, n_iter=2000, transform_y=(norm_y, norm_y_inv)),
    },

    'xgb4': {
        'features': ['numeric', 'categorical_dummy'],
        'model': Xgb({
            'max_depth': 7,
            'eta': 0.04,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'min_child_weight': 4,
        }, n_iter=2000, transform_y=(norm_y, norm_y_inv)),
    },

    'nn-tst': {
        'features': ['numeric'],
        'n_bags': 1,
        'model': Keras({'l1': 1e-3, 'l2': 1e-3, 'n_epoch': 1, 'batch_size': 128, 'layers': [10]}),
    },

    'nn1': {
        'features': ['numeric', 'categorical_encoded'],
        'n_bags': 1,
        'model': Keras({'l1': 1e-3, 'l2': 1e-3, 'n_epoch': 100, 'batch_size': 48, 'layers': [400, 200], 'dropouts': [0.4, 0.2]}),
    },

    'nn2': {
        'features': ['numeric_scaled', 'categorical_dummy'],
        'n_bags': 3,
        'model': Keras({'l1': 1e-5, 'l2': 1e-5, 'n_epoch': 50, 'batch_size': 128, 'layers': [400, 200], 'dropouts': [0.4, 0.2]}, scale=False),
    },

    'nn3': {
        'features': ['numeric'],
        'feature_builders': [CategoricalMeanEncoded(1000)],
        'n_bags': 1,
        'model': Keras({'l1': 1e-6, 'l2': 1e-6, 'n_epoch': 100, 'batch_size': 48, 'layers': [200, 100], 'dropouts': [0.2, 0.1]}),
    },

    'nn4': {
        'features': ['svd'],
        'n_bags': 1,
        'model': Keras({'l1': 1e-5, 'l2': 1e-5, 'n_epoch': 25, 'batch_size': 128, 'layers': [300, 100], 'dropouts': [0.3, 0.1]}),
    },

    'et1': {
        'features': ['numeric', 'categorical_encoded'],
        'model': Sklearn(ExtraTreesRegressor(50, max_features=0.2, n_jobs=-1), transform_y=(np.log, np.exp)),
    },

    'rf1': {
        'features': ['numeric', 'categorical_encoded'],
        'model': Sklearn(RandomForestRegressor(100, max_features=0.2, n_jobs=-1), transform_y=(np.log, np.exp)),
    },

    'lr1': {
        'features': ['numeric_scaled', 'categorical_dummy'],
        'model': Sklearn(Ridge(1e-3), transform_y=(np.log, np.exp)),
    },

    'lr2': {
        'features': ['numeric', 'categorical_encoded'],
        'model': Sklearn(Pipeline([('sc', StandardScaler(with_mean=False)), ('lr', Ridge(1e-3))]), transform_y=(np.log, np.exp)),
    },

    'lr2': {
        'features': ['svd'],
        'model': Sklearn(Ridge(1e-3), transform_y=(np.log, np.exp)),
    },

    'knn1': {
        'features': ['svd'],
        'model': Sklearn(Pipeline([('sc', StandardScaler(with_mean=False)), ('knn', KNeighborsRegressor(5))]), transform_y=(np.log, np.exp)),
    },

    'lsh1': {
        'features': ['numeric', 'categorical_encoded'],
        'model': LshForest(transform_y=(np.log, np.exp)),
    },

    'l2_nn': {
        'predictions': l1_predictions,
        'n_bags': 2,
        'n_splits': 2,
        'model': Keras({'l1': 1e-5, 'l2': 1e-5, 'lr': 1e-3, 'n_epoch': 7, 'batch_size': 128, 'layers': [10]}),
    },

    'l2_nn2': {
        'features': ['numeric_scaled', 'categorical_dummy'],
        'predictions': l1_predictions,
        'n_bags': 2,
        'n_splits': 2,
        'model': Keras({'l1': 1e-5, 'l2': 1e-5, 'lr': 1e-3, 'n_epoch': 700, 'batch_size': 128, 'layers': [50, 20]}),
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

print "Preset: %s" % args.preset

preset = presets[args.preset]

feature_builders = preset.get('feature_builders', [])

n_bags = preset.get('n_bags', 1)
n_splits = preset.get('n_splits', 1)

print "Loading train data..."
train_x = load_x('train', preset)
train_y = Dataset.load_part('train', 'loss')
train_p = pd.Series(0.0, index=Dataset.load_part('train', 'id'))
train_r = Dataset.load('train', parts=np.unique(sum([b.requirements for b in feature_builders], ['loss'])))

print "Loading test data..."
test_x = load_x('test', preset)
test_p = pd.Series(0.0, index=Dataset.load_part('test', 'id'))
test_r = Dataset.load('test', parts=np.unique([b.requirements for b in feature_builders]))

maes = []

for split in xrange(n_splits):
    print
    print "Training split %d..." % split

    for fold, (fold_train_idx, fold_eval_idx) in enumerate(KFold(len(train_y), n_folds, shuffle=True, random_state=2016 + 17*split)):
        print
        print "  Fold %d..." % fold

        fold_train_x = train_x[fold_train_idx]
        fold_train_y = train_y[fold_train_idx]
        fold_train_r = train_r.slice(fold_train_idx)

        fold_eval_x = train_x[fold_eval_idx]
        fold_eval_y = train_y[fold_eval_idx]
        fold_eval_r = train_r.slice(fold_eval_idx)

        fold_test_x = test_x
        fold_test_r = test_r

        if len(feature_builders) > 0:  # TODO: Move inside of bagging loop
            print "    Building per-fold features..."

            fold_train_x = [fold_train_x]
            fold_eval_x = [fold_eval_x]
            fold_test_x = [fold_test_x]

            for fb in feature_builders:
                fold_train_x.append(fb.fit_transform(fold_train_r))
                fold_eval_x.append(fb.transform(fold_eval_r))
                fold_test_x.append(fb.transform(fold_test_r))

            fold_train_x = hstack(fold_train_x)
            fold_eval_x = hstack(fold_eval_x)
            fold_test_x = hstack(fold_test_x)

        fold_eval_p = np.zeros((fold_eval_x.shape[0], ))
        fold_test_p = np.zeros((fold_test_x.shape[0], ))

        for i in xrange(n_bags):
            print "    Training model %d..." % i

            # Fit model
            model = preset['model']
            model.fit(fold_train_x, fold_train_y, fold_eval_x, fold_eval_y, seed=42 + 13*i + 29*split)

            print "    Predicting eval..."
            fold_eval_p += model.predict(fold_eval_x)

            print "    Predicting test..."
            fold_test_p += model.predict(fold_test_x)

        # Normalize train/test predictions
        fold_eval_p /= n_bags
        fold_test_p /= n_bags

        # Normalize train predictions
        train_p.iloc[fold_eval_idx] += fold_eval_p
        test_p += fold_test_p

        # Calculate err
        maes.append(mean_absolute_error(fold_eval_y, fold_eval_p))

        print "  MAE: %.5f" % maes[-1]

        # Free mem
        del fold_train_x, fold_train_y, fold_eval_x, fold_eval_y


## Analyzing predictions

test_p /= n_splits * n_folds
train_p /= n_splits

mae_mean = np.mean(maes)
mae_std = np.std(maes)
mae = mean_absolute_error(train_y, train_p)

print
print "CV MAE: %.5f +- %.5f" % (mae_mean, mae_std)
print "CV RES MAE: %.5f" % mae

name = "%s-%s-%.5f" % (datetime.datetime.now().strftime('%Y%m%d-%H%M'), args.preset, mae)

print
print "Saving predictions... (%s)" % name

for part, pred in [('train', train_p), ('test', test_p)]:
    pred.rename('loss', inplace=True)
    pred.index.rename('id', inplace=True)
    pred.to_csv('preds/%s-%s.csv' % (name, part), header=True)

copy2(os.path.realpath(__file__), os.path.join("preds", "%s-code.py" % name))

print "Done."
