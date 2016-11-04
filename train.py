import numpy as np
import pandas as pd
import xgboost as xgb
import scipy.sparse as sp

import argparse
import os
import datetime

from math import sqrt
from shutil import copy2, rmtree

np.random.seed(1337)

from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import KFold, train_test_split
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import LinearSVR
from sklearn.neighbors import LSHForest
from sklearn.model_selection import ParameterGrid
from sklearn.datasets import dump_svmlight_file
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Dense, Dropout, Lambda
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1l2
from keras.optimizers import SGD, Adam, Nadam, Adadelta
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from keras import backend as K
from keras import initializations
from keras_util import ExponentialMovingAverage, batch_generator

from pylightgbm.models import GBMRegressor

from scipy.stats import boxcox

from bayes_opt import BayesianOptimization

from util import Dataset, load_prediction, hstack


class CategoricalMeanEncoded(object):

    requirements = ['categorical']

    def __init__(self, C=100, loo=False, noisy=True, noise_std=None, random_state=11):
        self.random_state = np.random.RandomState(random_state)
        self.C = C
        self.loo = loo
        self.noisy = noisy
        self.noise_std = noise_std

    def fit_transform(self, ds):
        train_cat = ds['categorical']
        train_target = pd.Series(np.log(ds['loss']))
        train_res = np.zeros(train_cat.shape, dtype=np.float32)

        self.global_target_mean = train_target.mean()
        self.global_target_std = train_target.std() if self.noise_std is None else self.noise_std

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
            test_series = pd.Series(test_cat[:, col])

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

    def __init__(self, params, n_iter=400, transform_y=None, param_grid=None, huber=None, fair=None):
        self.params = self.default_params.copy()

        for k in params:
            self.params[k] = params[k]

        self.n_iter = n_iter
        self.transform_y = transform_y
        self.param_grid = param_grid
        self.huber = huber
        self.fair = fair

    def fit(self, X_train, y_train, X_eval=None, y_eval=None, seed=42):
        if self.transform_y is not None:
            y_tr, y_inv = self.transform_y

            y_train = y_tr(y_train)
            y_eval = y_tr(y_eval)

            feval = lambda y_pred, y_true: ('mae', mean_absolute_error(y_inv(y_true.get_label()), y_inv(y_pred)))
        else:
            feval = None

        if self.huber is not None:
            fobj = self.huber_approx_obj
        elif self.fair is not None:
            fobj = self.fair_obj
        else:
            fobj = None

        dtrain = xgb.DMatrix(X_train, label=y_train)
        deval = xgb.DMatrix(X_eval, label=y_eval)

        params = self.params.copy()
        params['seed'] = seed
        params['base_score'] = np.median(y_train)

        self.model = xgb.train(params, dtrain, self.n_iter, [(deval, 'eval'), (dtrain, 'train')], fobj, feval, verbose_eval=20)

    def predict(self, X):
        pred = self.model.predict(xgb.DMatrix(X))

        if self.transform_y is not None:
            _, y_inv = self.transform_y

            pred = y_inv(pred)

        return pred

    def optimize(self, X_train, y_train, X_eval, y_eval, seed=42):
        if self.transform_y is not None:
            y_tr, y_inv = self.transform_y

            y_train = y_tr(y_train)
            y_eval = y_tr(y_eval)

            feval = lambda y_pred, y_true: ('mae', mean_absolute_error(y_inv(y_true.get_label()), y_inv(y_pred)))
        else:
            feval = None

        dtrain = xgb.DMatrix(X_train, label=y_train)
        deval = xgb.DMatrix(X_eval, label=y_eval)

        for ps in ParameterGrid(self.param_grid):
            print "Using %s" % str(ps)

            params = self.params.copy()
            params['seed'] = seed

            for k in ps:
                params[k] = ps[k]

            model = xgb.train(params, dtrain, self.n_iter, [(deval, 'eval'), (dtrain, 'train')], objective, feval, verbose_eval=10)
            print "Result for %s: %.5f" % (str(ps), feval(model.predict(deval), deval)[1])

    def huber_approx_obj(self, preds, dtrain):
        d = preds - dtrain.get_label()

        scale = 1 + (d / self.huber) ** 2
        scale_sqrt = np.sqrt(scale)

        grad = d / scale_sqrt
        hess = 1 / scale / scale_sqrt

        return grad, hess

    def fair_obj(self, preds, dtrain):
        x = preds - dtrain.get_label()
        c = self.fair

        den = np.abs(x)+c

        grad = c*x / den
        hess = c*c / den ** 2

        return grad, hess


class LightGBM(object):

    default_params = {
        'exec_path': 'lightgbm',
        'num_threads': 4
    }

    def __init__(self, params, transform_y=None):
        self.params = self.default_params.copy()

        for k in params:
            self.params[k] = params[k]

        self.transform_y = transform_y

    def fit(self, X_train, y_train, X_eval=None, y_eval=None, seed=42):
        if self.transform_y is not None:
            y_tr, y_inv = self.transform_y

            y_train = y_tr(y_train)
            y_eval = y_tr(y_eval)

        params = self.params.copy()
        params['bagging_seed'] = seed
        params['feature_fraction_seed'] = seed + 3

        self.model = GBMRegressor(**params)
        self.model.fit(X_train, y_train, test_data=[(X_eval, y_eval)])

    def predict(self, X):
        pred = self.model.predict(X)

        if self.transform_y is not None:
            _, y_inv = self.transform_y

            pred = y_inv(pred)

        return pred


class LibFM(object):

    default_params = {
    }

    def __init__(self, params={}, transform_y=None):
        self.params = self.default_params.copy()

        for k in params:
            self.params[k] = params[k]

        self.transform_y = transform_y
        self.exec_path = 'libFM'
        self.tmp_dir = "libfm_models/{}".format(datetime.datetime.now().strftime('%Y%m%d-%H%M'))

    def __del__(self):
        #if os.path.exists(self.tmp_dir):
        #    rmtree(self.tmp_dir)
        pass

    def fit(self, X_train, y_train, X_eval=None, y_eval=None, seed=42):
        if self.transform_y is not None:
            y_tr, y_inv = self.transform_y

            y_train = y_tr(y_train)
            y_eval = y_tr(y_eval)

        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        train_file = os.path.join(self.tmp_dir, 'train.svm')
        eval_file = os.path.join(self.tmp_dir, 'eval.svm')
        out_file = os.path.join(self.tmp_dir, 'out.txt')

        print "Exporting train..."
        with open(train_file, 'w') as f:
            dump_svmlight_file(*shuffle(X_train, y_train, random_state=seed), f=f)

        print "Exporting eval..."
        with open(eval_file, 'w') as f:
            dump_svmlight_file(X_eval, y_eval, f=f)

        params = self.params.copy()
        params['seed'] = seed
        params['task'] = 'r'
        params['train'] = train_file
        params['test'] = eval_file
        params['out'] = out_file
        params['save_model'] = os.path.join(self.tmp_dir, 'model.libfm')
        params = " ".join("-{} {}".format(k, params[k]) for k in params)

        command = "{} {}".format(self.exec_path, params)

        print command
        os.system(command)

    def predict(self, X):
        train_file = os.path.join(self.tmp_dir, 'train.svm')
        pred_file = os.path.join(self.tmp_dir, 'pred.svm')
        out_file = os.path.join(self.tmp_dir, 'out.txt')

        print "Exporting pred..."
        with open(pred_file, 'w') as f:
            dump_svmlight_file(X, np.zeros(X.shape[0]), f=f)

        params = self.params.copy()
        params['iter'] = 0
        params['task'] = 'r'
        params['train'] = train_file
        params['test'] = pred_file
        params['out'] = out_file
        params['load_model'] = os.path.join(self.tmp_dir, 'model.libfm')
        params = " ".join("-{} {}".format(k, params[k]) for k in params)

        command = "{} {}".format(self.exec_path, params)

        print command
        os.system(command)

        pred = pd.read_csv(out_file, header=None).values.flatten()

        if self.transform_y is not None:
            _, y_inv = self.transform_y

            pred = y_inv(pred)

        return pred


class Sklearn(object):

    def __init__(self, model, transform_y=None, param_grid=None):
        self.model = model
        self.transform_y = transform_y
        self.param_grid = param_grid

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

    def optimize(self, X_train, y_train, X_eval, y_eval):
        def fun(**params):
            for k in params:
                if type(self.param_grid[k][0]) is int:
                    params[k] = int(params[k])

            print "Trying %s..." % str(params)

            self.model.set_params(**params)
            self.fit(X_train, y_train)
            pred = self.predict(X_eval)

            mae = mean_absolute_error(y_eval, pred)

            print "MAE: %.5f" % mae

            return -mae

        opt = BayesianOptimization(fun, self.param_grid)
        opt.maximize()

        print "Best mae: %.5f, params: %s" % (opt.res['max']['max_val'], opt.res['mas']['max_params'])


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

    def __init__(self, params, transform_y=None, scale=True, loss='mae'):
        self.params = params
        self.transform_y = transform_y
        self.scale = scale
        self.loss = loss

    def fit(self, X_train, y_train, X_eval=None, y_eval=None, seed=42):
        params = self.params

        if callable(params):
            params = params()

        np.random.seed(seed * 11 + 137)

        if self.transform_y is not None:
            y_tr, y_inv = self.transform_y

            y_train = y_tr(y_train)
            y_eval = y_tr(y_eval)

        if self.scale:
            self.scaler = StandardScaler(with_mean=False)

            X_train = self.scaler.fit_transform(X_train)
            X_eval = self.scaler.transform(X_eval)

        self.model = Sequential()

        for i, layer_size in enumerate(params['layers']):
            reg = l1l2(params['l1'], params['l2'])

            if i == 0:
                self.model.add(Dense(layer_size, init='he_normal', W_regularizer=reg, input_shape=(X_train.shape[1],)))
            else:
                self.model.add(Dense(layer_size, init='he_normal', W_regularizer=reg))

            #self.model.add(BatchNormalization())
            self.model.add(PReLU())

            if 'dropouts' in params:
                self.model.add(Dropout(params['dropouts'][i]))

        self.model.add(Dense(1, init='he_normal'))

        self.model.compile(optimizer=params.get('optimizer', 'adadelta'), loss=self.loss)

        self.model.fit_generator(
            generator=batch_generator(X_train, y_train, params['batch_size'], True), samples_per_epoch=X_train.shape[0],
            validation_data=batch_generator(X_eval, y_eval, 800), nb_val_samples=X_eval.shape[0],
            nb_epoch=params['n_epoch'], verbose=1, callbacks=params.get('callbacks', []))

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
    prediction_parts = [p.clip(lower=0.1).reshape((p.shape[0], 1)) for p in prediction_parts]

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
parser.add_argument('--optimize', action='store_true', help='optimize model params')
parser.add_argument('--fold', type=int, help='specify fold')
parser.add_argument('--threads', type=int, default=-1, help='specify thread count')


args = parser.parse_args()

Xgb.default_params['nthread'] = args.threads
LightGBM.default_params['num_threads'] = args.threads

n_folds = 8

l1_predictions = [

    #'20161027-1345-xgb-ce-1137.71847',
    #'20161027-1518-xgb-ce-2-1135.34796',

    #'20161027-1541-xgbf-ce-1139.27459',
    #'20161027-1606-xgbf-ce-2-1136.41797',

    #'20161027-1832-libfm-cd-1200.11201',

    #'20161027-2008-lgb-ce-1136.32506',

    '20161027-2048-lr-ce-1291.06963',
    '20161027-2110-lr-cd-1248.84696',
    '20161027-2111-lr-svd-1247.38512',
    '20161030-0044-lr-cd-nr-1248.64251',

    '20161027-2330-et-ce-1217.14724',
    '20161027-2340-rf-ce-1200.99200',

    '20161028-0031-gb-ce-1151.11060',

    '20161013-1512-xgb1-1146.11469',

    #'20161014-1330-xgb3-1143.31331',
    #'20161017-0645-xgb3-1137.53294',
    #'20161018-0434-xgb3-1137.17603',
    '20161027-0203-xgb3-1136.95146',

    '20161019-1805-xgb5-1138.26298',

    #'20161016-2155-nn2-1142.81539',
    #'20161017-0252-nn2-1138.89229',
    '20161018-1033-nn2-1138.11347',

    #'20161017-1350-nn4-1165.61897',

    #'20161019-1157-nn5-1142.70844',
    '20161019-2334-nn5-1142.50482',

    '20161028-1005-nn-svd-1144.31187',

    '20161026-1055-lgb1-1135.48359',

    #'20161021-2054-xgb7-1140.67644',
    '20161022-1736-xgb7-1138.66039',
    '20161027-1932-xgb-ce-2-1134.04010',
    '20161028-1420-xgb-ce-3-1132.58408',

    #'20161022-2023-xgbf1-1137.23245',
    '20161023-0643-xgbf1-1133.28725',

    '20161025-1849-xgbf3-1133.20314',

    '20161028-0039-xgbf-ce-2-1133.25472',
    '20161027-2321-xgbf-ce-3-1130.03141',
    '20161028-1909-xgbf-ce-4-1129.65181',

    '20161026-0127-libfm1-1195.55162',
    '20161028-1032-libfm-svd-1180.69290',
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

    'xgb-ce': {
        'features': ['numeric', 'categorical_encoded'],
        'n_bags': 2,
        'model': Xgb({
            'max_depth': 7,
            'eta': 0.03,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'min_child_weight': 2,
            'gamma': 0.2,
        }, n_iter=2000, transform_y=(norm_y, norm_y_inv)),
    },

    'xgb-ce-2': {
        'features': ['numeric', 'categorical_encoded'],
        'n_bags': 2,
        'model': Xgb({
            'max_depth': 12,
            'eta': 0.01,
            'colsample_bytree': 0.5,
            'subsample': 0.8,
            'gamma': 1,
            'alpha': 1,
        }, n_iter=3000, transform_y=(lambda x: np.log(x + 200), lambda x: np.maximum(np.exp(x) - 200, 0))),
    },

    'xgb-ce-3': {
        'features': ['numeric', 'categorical_encoded'],
        'n_bags': 2,
        'model': Xgb({
            'max_depth': 14,
            'eta': 0.01,
            'colsample_bytree': 0.5,
            'subsample': 0.8,
            'gamma': 1.5,
            'alpha': 1,
        }, n_iter=3000, transform_y=(lambda x: np.log(x + 200), lambda x: np.maximum(np.exp(x) - 200, 0))),
    },

    'xgb-ce-tst': {
        'features': ['numeric', 'categorical_encoded'],
        'n_bags': 2,
        'model': Xgb({
            'max_depth': 14,
            'eta': 0.01,
            'colsample_bytree': 0.5,
            'subsample': 0.8,
            'gamma': 1.5,
            'alpha': 1,
        }, n_iter=3000, transform_y=(lambda x: np.log(x + 200), lambda x: np.maximum(np.exp(x) - 200, 0))),
    },

    'xgb4': {
        'features': ['numeric', 'categorical_dummy'],
        'model': Xgb({
            'max_depth': 7,
            'eta': 0.02,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'min_child_weight': 2,
        }, n_iter=3000, transform_y=(norm_y, norm_y_inv)),
    },

    'xgb6': {
        'features': ['numeric', 'categorical_encoded'],
        'model': Xgb({
            'max_depth': 7,
            'eta': 0.03,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'min_child_weight': 4,
        }, n_iter=2000, transform_y=(norm_y, norm_y_inv), param_grid={'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]}),
    },

    'xgb7': {
        'features': ['numeric'],
        'feature_builders': [CategoricalMeanEncoded(C=10000, noisy=False, loo=False)],
        'model': Xgb({
            'max_depth': 7,
            'eta': 0.03,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'min_child_weight': 4,
        }, n_iter=2000, transform_y=(norm_y, norm_y_inv), param_grid={'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]}),
    },

    'xgbh-ce': {
        'features': ['numeric', 'categorical_encoded'],
        #'n_bags': 2,
        'model': Xgb({
            'max_depth': 7,
            'eta': 0.05,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'min_child_weight': 4,
        }, n_iter=2000, huber=100, param_grid={'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]}),
    },

    'xgbf-ce': {
        'features': ['numeric', 'categorical_encoded'],
        #'n_bags': 3,
        'model': Xgb({
            'max_depth': 7,
            'eta': 0.05,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'min_child_weight': 4,
            'alpha': 0.0005,
        }, n_iter=1100, fair=1, transform_y=(norm_y, norm_y_inv), param_grid={'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]}),
    },

    'xgbf-ce-2': {
        'features': ['numeric', 'categorical_encoded'],
        'n_bags': 2,
        'model': Xgb({
            'max_depth': 8,
            'eta': 0.04,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'gamma': 0.45,
            'alpha': 0.0005,
        }, n_iter=1320, fair=1, transform_y=(norm_y, norm_y_inv), param_grid={'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]}),
    },

    'xgbf-ce-3': {
        'features': ['numeric', 'categorical_encoded'],
        'n_bags': 2,
        'model': Xgb({
            'max_depth': 12,
            'eta': 0.04,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'gamma': 0.5,
            'alpha': 0.5,
        }, n_iter=2000, fair=1, transform_y=(norm_y, norm_y_inv), param_grid={'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]}),
    },

    'xgbf-ce-4': {
        'features': ['numeric', 'categorical_encoded'],
        'n_bags': 2,
        'model': Xgb({
            'max_depth': 15,
            'eta': 0.04,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'gamma': 0.6,
            'alpha': 0.5,
        }, n_iter=2700, fair=1, transform_y=(norm_y, norm_y_inv), param_grid={'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]}),
    },

    'xgbf-ce-4-2': {
        'features': ['numeric', 'categorical_encoded'],
        'n_bags': 2,
        'model': Xgb({
            'max_depth': 15,
            'eta': 0.04,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'gamma': 1.5,
            'alpha': 1.0,
        }, n_iter=2700, fair=1, transform_y=(np.log, np.exp), param_grid={'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]}),
    },

    'xgbf-ce-tst': {
        'features': ['numeric', 'categorical_encoded'],
        #'n_bags': 3,
        'model': Xgb({
            'max_depth': 18,
            'eta': 0.04,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'gamma': 0.6,
            'alpha': 0.6,
        }, n_iter=5000, fair=1, transform_y=(norm_y, norm_y_inv), param_grid={'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]}),
    },

    'xgbf-tst': {
        'features': ['numeric', 'categorical_encoded'],
        'n_bags': 3,
        'model': Xgb({
            'max_depth': 8,
            'eta': 0.05,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'gamma': 0.45,
            'alpha': 0.0005,
            #'lambda': 1.0,
        }, n_iter=1100, fair=1, transform_y=(norm_y, norm_y_inv), param_grid={'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]}),
    },

    'xgbf-cm-tst': {
        'features': ['numeric'],
        'feature_builders': [CategoricalMeanEncoded(C=10000, noisy=True, noise_std=0.1, loo=False)],
        'n_bags': 2,
        'model': Xgb({
            'max_depth': 7,
            'eta': 0.05,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'min_child_weight': 4,
            'alpha': 0.0005,
        }, n_iter=1100, fair=1, transform_y=(norm_y, norm_y_inv), param_grid={'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]}),
    },

    'lgb-ce': {
        'features': ['numeric', 'categorical_encoded'],
        'n_bags': 2,
        'model': LightGBM({
            'num_iterations': 5500,
            'learning_rate': 0.004,
            'num_leaves': 250,
            'min_data_in_leaf': 4,
            'feature_fraction': 0.3,
            'bagging_fraction': 0.95,
            'bagging_freq': 5,
            'metric_freq': 10
        }, transform_y=(norm_y, norm_y_inv)),
    },

    'lgb-ce-tst': {
        'features': ['numeric', 'categorical_encoded'],
        'n_bags': 2,
        'model': LightGBM({
            'num_iterations': 5500,
            'learning_rate': 0.004,
            'num_leaves': 250,
            'min_data_in_leaf': 4,
            'feature_fraction': 0.3,
            'bagging_fraction': 0.95,
            'bagging_freq': 5,
            'metric_freq': 10
        }, transform_y=(norm_y, norm_y_inv)),
    },

    'lgb-cm-tst': {
        'features': ['numeric'],
        'feature_builders': [CategoricalMeanEncoded(C=10000, noisy=True, noise_std=0.1, loo=False)],
        #'n_bags': 2,
        'model': LightGBM({
            'num_iterations': 4000,
            'learning_rate': 0.006,
            'num_leaves': 250,
            'min_data_in_leaf': 2,
            'feature_fraction': 0.25,
            'bagging_fraction': 0.95,
            'bagging_freq': 5,
            'metric_freq': 10
        }, transform_y=(norm_y, norm_y_inv)),
    },

    'libfm-cd': {
        'features': ['numeric_scaled', 'categorical_dummy'],
        'model': LibFM(params={
            'method': 'sgd',
            'learn_rate': 0.0001,
            'iter': 200,
            'dim': '1,1,12',
            'regular': '0,0,0.0002'
        }, transform_y=(np.log, np.exp)),
    },

    'libfm-svd': {
        'features': ['svd'],
        'model': LibFM(params={
            'method': 'sgd',
            'learn_rate': 0.00007,
            'iter': 200,
            'dim': '1,1,12',
            'regular': '0,0,0.0002'
        }, transform_y=(np.log, np.exp)),
    },

    'nn-tst': {
        'features': ['numeric'],
        'model': Keras({'l1': 1e-3, 'l2': 1e-3, 'n_epoch': 1, 'batch_size': 128, 'layers': [10]}),
    },

    'nn1': {
        'features': ['numeric', 'categorical_encoded'],
        'model': Keras({'l1': 1e-3, 'l2': 1e-3, 'n_epoch': 100, 'batch_size': 48, 'layers': [400, 200], 'dropouts': [0.4, 0.2]}),
    },

    'nn-cd': {
        'features': ['numeric_scaled', 'categorical_dummy'],
        'n_bags': 2,
        'model': Keras(lambda: {'l1': 1e-5, 'l2': 1e-5, 'n_epoch': 80, 'batch_size': 128, 'layers': [400, 200], 'dropouts': [0.4, 0.2], 'optimizer': Adadelta(), 'callbacks': [ExponentialMovingAverage()]}, scale=False),
    },

    'nn-cd-2': {
        'features': ['numeric_scaled', 'categorical_dummy'],
        'n_bags': 2,
        'model': Keras(lambda: {'l1': 1e-5, 'l2': 1e-5, 'n_epoch': 80, 'batch_size': 128, 'layers': [400, 200], 'dropouts': [0.4, 0.2], 'optimizer': Adadelta(), 'callbacks': [ExponentialMovingAverage()]}, scale=False, transform_y=(np.log, np.exp)),
    },

    'nn-cd-3': {
        'features': ['numeric_scaled', 'categorical_dummy'],
        #'n_bags': 2,
        'model': Keras(lambda: {'l1': 1e-6, 'l2': 1e-6, 'n_epoch': 20, 'batch_size': 128, 'layers': [300, 200], 'dropouts': [0.3, 0.2], 'optimizer': SGD(1e-5, momentum=0.8, nesterov=True, decay=1e-5), 'callbacks': [ExponentialMovingAverage()]}, scale=False, transform_y=(np.log, np.exp), loss='mse'),
    },

    'nn-cd-4': {
        'features': ['numeric_rank_norm', 'categorical_dummy'],
        'n_bags': 2,
        'model': Keras(lambda: {'l1': 1e-5, 'l2': 1e-5, 'n_epoch': 80, 'batch_size': 128, 'layers': [400, 200], 'dropouts': [0.4, 0.2], 'optimizer': Adadelta(), 'callbacks': [ExponentialMovingAverage()]}, scale=False),
    },

    'nn-svd': {
        'features': ['svd'],
        'n_bags': 2,
        'model': Keras(lambda: {'l1': 1e-5, 'l2': 1e-5, 'n_epoch': 80, 'batch_size': 128, 'layers': [400, 200], 'dropouts': [0.4, 0.2], 'optimizer': Adadelta(), 'callbacks': [ExponentialMovingAverage()]}, scale=False),
    },

    'nn3': {
        'features': ['numeric'],
        'feature_builders': [CategoricalMeanEncoded(1000)],
        'model': Keras(lambda: {'l1': 1e-5, 'l2': 1e-5, 'n_epoch': 80, 'batch_size': 128, 'layers': [400, 200], 'dropouts': [0.4, 0.2], 'optimizer': Adadelta(), 'callbacks': [ExponentialMovingAverage()]}, scale=True),
    },

    'nn4': {
        'n_bags': 2,
        'features': ['svd'],
        'model': Keras(lambda: {'l1': 1e-5, 'l2': 1e-5, 'n_epoch': 80, 'batch_size': 128, 'layers': [400, 200], 'dropouts': [0.4, 0.2], 'optimizer': Adadelta(), 'callbacks': [ExponentialMovingAverage()]}, scale=True),
    },

    'nn5': {
        'n_bags': 2,
        'features': ['numeric_scaled', 'categorical_dummy'],
        'model': Keras(lambda: {'l1': 1e-5, 'l2': 1e-5, 'n_epoch': 70, 'batch_size': 128, 'layers': [400, 200], 'dropouts': [0.4, 0.2], 'optimizer': Adam(decay=1e-5), 'callbacks': [ReduceLROnPlateau(patience=10, factor=0.2, cooldown=5), ExponentialMovingAverage()]}, scale=False),
    },

    'gb-ce': {
        'features': ['numeric', 'categorical_encoded'],
        'model': Sklearn(GradientBoostingRegressor(loss='lad', n_estimators=300, max_depth=7, max_features=0.2), param_grid={'n_estimators': (200, 400), 'max_depth': (6, 8), 'max_features': (0.1, 0.4)}),
    },

    'et-ce': {
        'features': ['numeric', 'categorical_encoded'],
        'model': Sklearn(ExtraTreesRegressor(50, max_features=0.2, n_jobs=-1), transform_y=(np.log, np.exp)),
    },

    'rf-ce': {
        'features': ['numeric', 'categorical_encoded'],
        'model': Sklearn(RandomForestRegressor(100, max_features=0.2, n_jobs=-1), transform_y=(np.log, np.exp)),
    },

    'lr-cd': {
        'features': ['numeric_scaled', 'categorical_dummy'],
        'model': Sklearn(Ridge(1e-3), transform_y=(np.log, np.exp), param_grid={'C': (1e-3, 1e3)}),
    },

    'lr-cd-nr': {
        'features': ['numeric_rank_norm', 'categorical_dummy'],
        'model': Sklearn(Ridge(1e-3), transform_y=(np.log, np.exp), param_grid={'C': (1e-3, 1e3)}),
    },

    'lr-ce': {
        'features': ['numeric', 'categorical_encoded'],
        'model': Sklearn(Pipeline([('sc', StandardScaler(with_mean=False)), ('lr', Ridge(1e-3))]), transform_y=(np.log, np.exp)),
    },

    'lr-svd': {
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

    'l2-nn': {
        'predictions': l1_predictions,
        'n_bags': 2,
        'model': Keras(lambda: {'l1': 1e-5, 'l2': 1e-5, 'n_epoch': 30, 'batch_size': 128, 'layers': [50], 'dropouts': [0.1], 'optimizer': SGD(1e-3, momentum=0.8, nesterov=True, decay=3e-5), 'callbacks': [ReduceLROnPlateau(patience=5, factor=0.2, cooldown=3), ExponentialMovingAverage()]}),
    },

    'l2-nn2': {
        'features': ['categorical_dummy'],
        'predictions': l1_predictions,
        'n_bags': 2,
        'n_splits': 2,
        'model': Keras({'l1': 1e-5, 'l2': 1e-5, 'n_epoch': 700, 'batch_size': 128, 'layers': [400, 100]}),
    },

    'l2-nn-tst': {
        'predictions': l1_predictions,
        'n_bags': 2,
        'model': Keras(lambda: {'l1': 1e-4, 'l2': 1e-5, 'n_epoch': 30, 'batch_size': 128, 'layers': [100], 'dropouts': [0.4], 'optimizer': SGD(4e-3, momentum=0.8, nesterov=True, decay=3e-4), 'callbacks': [ReduceLROnPlateau(patience=5, factor=0.2, cooldown=3), ExponentialMovingAverage()]}),
    },

    'l2-lr': {
        'predictions': l1_predictions,
        'prediction_transform': np.log,
        'model': Sklearn(Ridge(), transform_y=(np.log, np.exp)),
    },

    'l2-xgbf': {
        'predictions': l1_predictions,
        'model': Xgb({
            'max_depth': 8,
            'eta': 0.03,
            'colsample_bytree': 0.3,
            'subsample': 0.5,
            'gamma': 0.01,
            'alpha': 0.0005,
            #'min_child_weight': 5,
        }, n_iter=5000, fair=1, transform_y=(np.log, np.exp)),
    },

    'l3-nn': {
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


if args.optimize:
    opt_train_idx, opt_eval_idx = train_test_split(range(len(train_y)), test_size=0.2)

    opt_train_x = train_x[opt_train_idx]
    opt_train_y = train_y[opt_train_idx]
    opt_train_r = train_r.slice(opt_train_idx)

    opt_eval_x = train_x[opt_eval_idx]
    opt_eval_y = train_y[opt_eval_idx]
    opt_eval_r = train_r.slice(opt_eval_idx)

    if len(feature_builders) > 0:  # TODO: Move inside of bagging loop
        print "    Building per-fold features..."

        opt_train_x = [opt_train_x]
        opt_eval_x = [opt_eval_x]

        for fb in feature_builders:
            opt_train_x.append(fb.fit_transform(opt_train_r))
            opt_eval_x.append(fb.transform(opt_eval_r))

        opt_train_x = hstack(opt_train_x)
        opt_eval_x = hstack(opt_eval_x)

    preset['model'].optimize(opt_train_x, opt_train_y, opt_eval_x, opt_eval_y)


print "Loading test data..."
test_x = load_x('test', preset)
test_p = pd.Series(0.0, index=Dataset.load_part('test', 'id'))
test_r = Dataset.load('test', parts=np.unique([b.requirements for b in feature_builders]))

maes = []

for split in xrange(n_splits):
    print
    print "Training split %d..." % split

    for fold, (fold_train_idx, fold_eval_idx) in enumerate(KFold(len(train_y), n_folds, shuffle=True, random_state=2016 + 17*split)):
        if args.fold is not None and fold != args.fold:
            continue

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
