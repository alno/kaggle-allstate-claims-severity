import numpy as np
import pandas as pd
import xgboost as xgb
import scipy.sparse as sp

import argparse
import os
import datetime
import itertools

from math import sqrt
from shutil import copy2, rmtree

np.random.seed(1337)

from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import KFold, train_test_split
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import LinearSVR
from sklearn.neighbors import LSHForest
from sklearn.model_selection import ParameterGrid
from sklearn.datasets import dump_svmlight_file
from sklearn.utils import shuffle

from sklearn_util import MedianExtraTreesRegressor

from keras.models import Sequential
from keras.layers import Dense, MaxoutDense, Dropout, Lambda
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, Nadam, Adadelta
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from keras import backend as K
from keras import regularizers
from keras_util import ExponentialMovingAverage, batch_generator

from pylightgbm.models import GBMRegressor

from scipy.stats import boxcox

from bayes_opt import BayesianOptimization

from util import Dataset, load_prediction, hstack


categoricals = Dataset.get_part_features('categorical')


class CategoricalAlphaEncoded(object):

    requirements = ['categorical']

    def __init__(self, combinations=[]):
        self.combinations = [map(categoricals.index, comb) for comb in combinations]

    def fit_transform(self, ds):
        return self.transform(ds)

    def transform(self, ds):
        test_cat = ds['categorical']
        test_res = np.zeros((test_cat.shape[0], len(categoricals) + len(self.combinations)), dtype=np.float32)

        for col in xrange(len(categoricals)):
            test_res[:, col] = self.transform_column(test_cat[:, col])

        for idx, comb in enumerate(self.combinations):
            col = idx + len(categoricals)
            test_res[:, col] = self.transform_column(map(''.join, test_cat[:, comb]))

        return test_res

    def transform_column(self, arr):
        def encode(charcode):
            r = 0
            ln = len(charcode)
            for i in range(ln):
                r += (ord(charcode[i])-ord('A')+1)*26**(ln-i-1)
            return r

        return np.array(map(encode, arr))

    def get_feature_names(self):
        return categoricals + ['_'.join(categoricals[c] for c in comb) for comb in self.combinations]


class CategoricalMeanEncoded(object):

    requirements = ['categorical']

    def __init__(self, C=100, loo=False, noisy=True, noise_std=None, random_state=11, combinations=[]):
        self.random_state = np.random.RandomState(random_state)
        self.C = C
        self.loo = loo
        self.noisy = noisy
        self.noise_std = noise_std
        self.combinations = [map(categoricals.index, comb) for comb in combinations]

    def fit_transform(self, ds):
        train_cat = ds['categorical']
        train_target = pd.Series(np.log(ds['loss'] + 100))
        train_res = np.zeros((train_cat.shape[0], len(categoricals) + len(self.combinations)), dtype=np.float32)

        self.global_target_mean = train_target.mean()
        self.global_target_std = train_target.std() if self.noise_std is None else self.noise_std

        self.target_sums = {}
        self.target_cnts = {}

        for col in xrange(len(categoricals)):
            train_res[:, col] = self.fit_transform_column(col, train_target, pd.Series(train_cat[:, col]))

        for idx, comb in enumerate(self.combinations):
            col = idx + len(categoricals)
            train_res[:, col] = self.fit_transform_column(col, train_target, pd.Series(map(''.join, train_cat[:, comb])))

        return train_res

    def transform(self, ds):
        test_cat = ds['categorical']
        test_res = np.zeros((test_cat.shape[0], len(categoricals) + len(self.combinations)), dtype=np.float32)

        for col in xrange(len(categoricals)):
            test_res[:, col] = self.transform_column(col, pd.Series(test_cat[:, col]))

        for idx, comb in enumerate(self.combinations):
            col = idx + len(categoricals)
            test_res[:, col] = self.transform_column(col, pd.Series(map(''.join, test_cat[:, comb])))

        return test_res

    def fit_transform_column(self, col, train_target, train_series):
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

        return np.exp(train_res_num / train_res_den).values

    def transform_column(self, col, test_series):
        test_res_num = test_series.map(self.target_sums[col]).fillna(0.0) + self.global_target_mean * self.C
        test_res_den = test_series.map(self.target_cnts[col]).fillna(0.0) + self.C

        return np.exp(test_res_num / test_res_den).values

    def get_feature_names(self):
        return categoricals + ['_'.join(categoricals[c] for c in comb) for comb in self.combinations]


class BaseAlgo(object):

    def fit_predict(self, train, val=None, test=None, **kwa):
        self.fit(train[0], train[1], val[0] if val else None, val[1] if val else None, **kwa)

        val_pred = self.predict(val[0])
        test_pred = self.predict(test[0])

        return val_pred, test_pred


class Xgb(BaseAlgo):

    default_params = {
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'silent': 1,
        'nthread': -1,
    }

    def __init__(self, params, n_iter=400, transform_y=None, param_grid=None, huber=None, fair=None, fair_decay=0):
        self.params = self.default_params.copy()

        for k in params:
            self.params[k] = params[k]

        self.n_iter = n_iter
        self.transform_y = transform_y
        self.param_grid = param_grid
        self.huber = huber
        self.fair = fair
        self.fair_decay = fair_decay

        if self.huber is not None:
            self.objective = self.huber_approx_obj
        elif self.fair is not None:
            self.objective = self.fair_obj
        else:
            self.objective = None

    def fit(self, X_train, y_train, X_eval=None, y_eval=None, seed=42, feature_names=None):
        if self.transform_y is not None:
            y_tr, y_inv = self.transform_y

            y_train = y_tr(y_train)
            y_eval = y_tr(y_eval)

            feval = lambda y_pred, y_true: ('mae', mean_absolute_error(y_inv(y_true.get_label()), y_inv(y_pred)))
        else:
            feval = None

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        deval = xgb.DMatrix(X_eval, label=y_eval, feature_names=feature_names)

        params = self.params.copy()
        params['seed'] = seed
        params['base_score'] = np.median(y_train)

        self.iter = 0
        self.model = xgb.train(params, dtrain, self.n_iter, [(deval, 'eval'), (dtrain, 'train')], self.objective, feval, verbose_eval=20)
        self.feature_names = feature_names

        print "    Feature importances: %s" % ', '.join('%s: %d' % t for t in sorted(self.model.get_fscore().items(), key=lambda t: -t[1]))

    def predict(self, X):
        pred = self.model.predict(xgb.DMatrix(X, feature_names=self.feature_names))

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

        def fun(**kw):
            params = self.params.copy()
            params['seed'] = seed
            params['base_score'] = np.median(y_train)

            for k in kw:
                if type(self.param_grid[k][0]) is int:
                    params[k] = int(kw[k])
                else:
                    params[k] = kw[k]

            print "Trying %s..." % str(params)

            self.iter = 0

            model = xgb.train(params, dtrain, 10000, [(dtrain, 'train'), (deval, 'eval')], self.objective, feval, verbose_eval=20, early_stopping_rounds=100)

            print "Score %.5f at iteration %d" % (model.best_score, model.best_iteration)

            return - model.best_score

        opt = BayesianOptimization(fun, self.param_grid)
        opt.maximize(n_iter=100)

        print "Best mae: %.5f, params: %s" % (opt.res['max']['max_val'], opt.res['mas']['max_params'])

    def huber_approx_obj(self, preds, dtrain):
        d = preds - dtrain.get_label()
        h = self.huber

        scale = 1 + (d / h) ** 2
        scale_sqrt = np.sqrt(scale)

        grad = d / scale_sqrt
        hess = 1 / scale / scale_sqrt

        return grad, hess

    def fair_obj(self, preds, dtrain):
        x = preds - dtrain.get_label()
        c = self.fair

        den = np.abs(x) * np.exp(self.fair_decay * self.iter) + c

        grad = c*x / den
        hess = c*c / den ** 2

        self.iter += 1

        return grad, hess


class LightGBM(BaseAlgo):

    default_params = {
        'exec_path': 'lightgbm',
        'num_threads': 4
    }

    def __init__(self, params, transform_y=None):
        self.params = self.default_params.copy()

        for k in params:
            self.params[k] = params[k]

        self.transform_y = transform_y

    def fit(self, X_train, y_train, X_eval=None, y_eval=None, seed=42, feature_names=None):
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


class LibFM(BaseAlgo):

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

    def fit(self, X_train, y_train, X_eval=None, y_eval=None, seed=42, feature_names=None):
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


class Sklearn(BaseAlgo):

    def __init__(self, model, transform_y=None, param_grid=None):
        self.model = model
        self.transform_y = transform_y
        self.param_grid = param_grid

    def fit(self, X_train, y_train, X_eval=None, y_eval=None, seed=42, feature_names=None):
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
        opt.maximize(n_iter=100)

        print "Best mae: %.5f, params: %s" % (opt.res['max']['max_val'], opt.res['mas']['max_params'])


class LshForest(BaseAlgo):

    def __init__(self, transform_y=None):
        self.transform_y = transform_y

    def fit(self, X_train, y_train, X_eval=None, y_eval=None, seed=42, feature_names=None):
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


class Keras(BaseAlgo):

    def __init__(self, arch, params, transform_y=None, scale=True, loss='mae'):
        self.arch = arch
        self.params = params
        self.transform_y = transform_y
        self.scale = scale
        self.loss = loss

    def fit(self, X_train, y_train, X_eval=None, y_eval=None, seed=42, feature_names=None):
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

        self.model = self.arch((X_train.shape[1],), params)
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


def regularizer(params):
    if 'l1' in params and 'l2' in params:
        return regularizers.l1l2(params['l1'], params['l2'])
    elif 'l1' in params:
        return regularizers.l1(params['l1'])
    elif 'l2' in params:
        return regularizers.l2(params['l2'])
    else:
        return None


def nn_lr(input_shape, params):
    model = Sequential()
    model.add(Dense(1, input_shape=input_shape))

    return model


def nn_mlp(input_shape, params):
    model = Sequential()

    for i, layer_size in enumerate(params['layers']):
        reg = regularizer(params)

        if i == 0:
            model.add(Dense(layer_size, init='he_normal', W_regularizer=reg, input_shape=input_shape))
        else:
            model.add(Dense(layer_size, init='he_normal', W_regularizer=reg))

        if params.get('batch_norm', False):
            model.add(BatchNormalization())

        if 'dropouts' in params:
            model.add(Dropout(params['dropouts'][i]))

        model.add(PReLU())

    model.add(Dense(1, init='he_normal'))

    return model


def nn_mlp_2(input_shape, params):
    model = Sequential()

    for i, layer_size in enumerate(params['layers']):
        reg = regularizer(params)

        if i == 0:
            model.add(Dense(layer_size, init='he_normal', W_regularizer=reg, input_shape=input_shape))
        else:
            model.add(Dense(layer_size, init='he_normal', W_regularizer=reg))

        model.add(PReLU())

        if params.get('batch_norm', False):
            model.add(BatchNormalization())

        if 'dropouts' in params:
            model.add(Dropout(params['dropouts'][i]))

    model.add(Dense(1, init='he_normal'))

    return model


def nn_maxout(input_shape, params):
    model = Sequential()

    for i, layer_size in enumerate(params['layers']):
        reg = regularizer(params)

        if i == 0:
            model.add(MaxoutDense(layer_size, init='he_normal', W_regularizer=reg, input_shape=input_shape))
        else:
            model.add(MaxoutDense(layer_size, init='he_normal', W_regularizer=reg))

#        if params.get('batch_norm', False):
#            model.add(BatchNormalization())

        if 'dropouts' in params:
            model.add(Dropout(params['dropouts'][i]))

    model.add(Dense(1, init='he_normal'))

    return model


def load_x(ds, preset):
    feature_parts = [Dataset.load_part(ds, part) for part in preset.get('features', [])]
    prediction_parts = [load_prediction(ds, p) for p in preset.get('predictions', [])]
    prediction_parts = [p.clip(lower=0.1).values.reshape((p.shape[0], 1)) for p in prediction_parts]

    if 'prediction_transform' in preset:
        prediction_parts = map(preset['prediction_transform'], prediction_parts)

    return hstack(feature_parts + prediction_parts)


def extract_feature_names(preset):
    x = []

    for part in preset.get('features', []):
        x += Dataset.get_part_features(part)

    for pred in preset.get('predictions', []):
        x.append(pred)

    return x


norm_y_lambda = 0.7


def norm_y(y):
    return boxcox(np.log1p(y), lmbda=norm_y_lambda)


def norm_y_inv(y_bc):
    return np.expm1((y_bc * norm_y_lambda + 1)**(1/norm_y_lambda))


y_norm = (norm_y, norm_y_inv)
y_log = (np.log, np.exp)


def y_log_ofs(ofs):
    def transform(y):
        return np.log(y + ofs)

    def inv_transform(yl):
        return np.clip(np.exp(yl) - ofs, 1.0, np.inf)

    return transform, inv_transform


## Main part


parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('preset', type=str, help='model preset (features and hyperparams)')
parser.add_argument('--optimize', action='store_true', help='optimize model params')
parser.add_argument('--fold', type=int, help='specify fold')
parser.add_argument('--threads', type=int, default=4, help='specify thread count')


args = parser.parse_args()

Xgb.default_params['nthread'] = args.threads
LightGBM.default_params['num_threads'] = args.threads

n_folds = 8


l1_predictions = [
    '20161027-2048-lr-ce-1291.06963',
    '20161112-1927-lr-cd-1247.90719',
    '20161112-2028-lr-svd-1248.10532',
    '20161030-0044-lr-cd-nr-1248.64251',
    '20161119-2107-lr-clrbf-1212.10289',
    '20161120-1710-lr-svd-clrbf-2-1205.77612',

    '20161027-2330-et-ce-1217.14724',
    '20161104-1322-et-ce-2-1214.13643',
    '20161104-1432-et-ce-3-1199.82233',
    '20161114-0336-et-ce-4-1194.75138',

    '20161027-2340-rf-ce-1200.99200',
    '20161104-1604-rf-ce-2-1193.61802',
    '20161114-0459-rf-ce-3-1190.27838',

    '20161028-0031-gb-ce-1151.11060',

    '20161013-1512-xgb1-1146.11469',
    '20161027-0203-xgb3-1136.95146',
    '20161019-1805-xgb5-1138.26298',

    '20161018-1033-nn2-1138.11347',
    '20161019-2334-nn5-1142.50482',

    '20161105-1053-nn-cd-2-1135.09238',
    '20161112-1903-nn-cd-3-1133.89751',
    '20161114-0046-nn-cd-3-1132.74636',

    '20161120-1618-nn-cd-clrbf-1132.30517',
    '20161122-0610-nn-cd-clrbf-2-1131.97963',

    '20161028-1005-nn-svd-1144.31187',

    '20161026-1055-lgb1-1135.48359',

    '20161112-0120-lgb-cd-1-1134.15660',
    '20161112-0551-lgb-cd-2-1132.30663',

    '20161022-1736-xgb7-1138.66039',
    '20161027-1932-xgb-ce-2-1134.04010',
    '20161028-1420-xgb-ce-3-1132.58408',

    '20161023-0643-xgbf1-1133.28725',
    '20161025-1849-xgbf3-1133.20314',

    '20161028-0039-xgbf-ce-2-1133.25472',
    '20161027-2321-xgbf-ce-3-1130.03141',
    '20161028-1909-xgbf-ce-4-1129.65181',
    '20161105-2104-xgbf-ce-5-1138.75039',
    '20161113-1944-xgbf-ce-6-1129.17629',
    '20161114-0641-xgbf-ce-7-1127.24376',
    '20161115-0812-xgbf-ce-8-1124.35470',
    '20161117-0948-xgbf-ce-8-1124.23084',

    '20161119-2149-xgbf-ce-clrbf-1-1151.82274',
    '20161120-1506-xgbf-ce-clrbf-2-1140.36323',

    '20161026-0127-libfm1-1195.55162',
    '20161028-1032-libfm-svd-1180.69290',
    '20161122-2028-libfm-svd-1179.83001',
]

l2_predictions = [
    '20161112-2136-l2-lr-1124.72277',
    #'20161113-0020-l2-et-1124.59154',
    #'20161112-1624-l2-nn-1120.42637',
    '20161113-0206-l2-nn-1119.67630',
    '20161115-1523-l2-nn-1118.50030',
    '20161112-2302-l2-xgbf-1121.74439',
    '20161115-0356-l2-xgbf-1120.47804',
]

presets = {
    'xgb-tst': {
        'features': ['numeric'],
        'model': Xgb({'max_depth': 5, 'eta': 0.05}, n_iter=10, param_grid={'colsample_bytree': [0.2, 1.0]}),
    },

    'xgb2': {
        'features': ['numeric', 'categorical_counts'],
        'model': Xgb({
            'max_depth': 7,
            'eta': 0.1,
            'colsample_bytree': 0.5,
            'subsample': 0.95,
            'min_child_weight': 5,
        }, n_iter=400, transform_y=y_norm, param_grid={'colsample_bytree': [0.2, 1.0]}),
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
        }, n_iter=2000, transform_y=y_norm),
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
        }, n_iter=3000, transform_y=y_norm),
    },

    'xgb6': {
        'features': ['numeric', 'categorical_encoded'],
        'model': Xgb({
            'max_depth': 7,
            'eta': 0.03,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'min_child_weight': 4,
        }, n_iter=2000, transform_y=y_norm, param_grid={'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]}),
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
        }, n_iter=2000, transform_y=y_norm, param_grid={'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]}),
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
        }, n_iter=1100, fair=1, transform_y=y_norm, param_grid={'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]}),
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
        }, n_iter=1320, fair=1, transform_y=y_norm, param_grid={'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]}),
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
        }, n_iter=2000, fair=1, transform_y=y_norm, param_grid={'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]}),
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
        }, n_iter=2700, fair=1, transform_y=y_norm, param_grid={'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]}),
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
        }, n_iter=2700, fair=1, transform_y=y_log, param_grid={'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]}),
    },

    'xgbf-ce-5': {
        'features': ['numeric', 'categorical_encoded'],
        #'n_bags': 3,
        'model': Xgb({
            'max_depth': 9,
            'eta': 0.04,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'alpha': 0.9,
        }, n_iter=1250, fair=150, fair_decay=0.001),
    },

    'xgbf-ce-6': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalAlphaEncoded(
                combinations=[('cat103', 'cat111'), ('cat2', 'cat6'), ('cat87', 'cat11'), ('cat103', 'cat4'), ('cat80', 'cat103'), ('cat73', 'cat82'), ('cat12', 'cat72'), ('cat80', 'cat12'), ('cat111', 'cat5'), ('cat2', 'cat111'), ('cat80', 'cat57'), ('cat80', 'cat79'), ('cat1', 'cat82'), ('cat11', 'cat13')]
            )],
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 13,
            'eta': 0.04,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'gamma': 0.6,
            'alpha': 0.5,
        }, n_iter=2700, fair=1, transform_y=y_norm, param_grid={'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]}),
    },

    'xgbf-ce-7': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalAlphaEncoded(
                combinations=[('cat103', 'cat111'), ('cat2', 'cat6'), ('cat87', 'cat11'), ('cat103', 'cat4'), ('cat80', 'cat103'), ('cat73', 'cat82'), ('cat12', 'cat72'), ('cat80', 'cat12'), ('cat111', 'cat5'), ('cat2', 'cat111'), ('cat80', 'cat57'), ('cat80', 'cat79'), ('cat1', 'cat82'), ('cat11', 'cat13')]
            )],
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 13,
            'eta': 0.04,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'gamma': 1.2,
            'alpha': 1.0,
        }, n_iter=2500, fair=1, transform_y=y_log_ofs(200), param_grid={'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]}),
    },

    'xgbf-ce-8': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalAlphaEncoded(
                combinations=[('cat103', 'cat111'), ('cat2', 'cat6'), ('cat87', 'cat11'), ('cat103', 'cat4'), ('cat80', 'cat103'), ('cat73', 'cat82'), ('cat12', 'cat72'), ('cat80', 'cat12'), ('cat111', 'cat5'), ('cat2', 'cat111'), ('cat80', 'cat57'), ('cat80', 'cat79'), ('cat1', 'cat82'), ('cat11', 'cat13')]
            )],
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 13,
            'eta': 0.01,
            'colsample_bytree': 0.2,
            'subsample': 0.95,
            'gamma': 1.15,
            'alpha': 1.0,
        }, n_iter=16000, fair=1, transform_y=y_log_ofs(200), param_grid={'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]}),
    },

    'xgbf-ce-9': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalAlphaEncoded(
                combinations=[('cat103', 'cat111'), ('cat2', 'cat6'), ('cat87', 'cat11'), ('cat103', 'cat4'), ('cat80', 'cat103'), ('cat73', 'cat82'), ('cat12', 'cat72'), ('cat80', 'cat12'), ('cat111', 'cat5'), ('cat2', 'cat111'), ('cat80', 'cat57'), ('cat80', 'cat79'), ('cat1', 'cat82'), ('cat11', 'cat13'), ('cat79', 'cat81'), ('cat81', 'cat13'), ('cat9', 'cat73'), ('cat2', 'cat81'), ('cat80', 'cat111'), ('cat79', 'cat111'), ('cat72', 'cat1'), ('cat23', 'cat103'), ('cat89', 'cat13'), ('cat57', 'cat14'), ('cat80', 'cat81'), ('cat81', 'cat11'), ('cat9', 'cat103'), ('cat23', 'cat36')]
            )],
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 12,
            'eta': 0.02,
            'colsample_bytree': 0.2,
            'subsample': 0.95,
            'gamma': 1.1,
            'alpha': 0.95,
        }, n_iter=8000, fair=1, transform_y=y_log_ofs(200), param_grid={'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]}),
    },

    'xgbf-ce-tst': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalAlphaEncoded(
                combinations=[('cat103', 'cat111'), ('cat2', 'cat6'), ('cat87', 'cat11'), ('cat103', 'cat4'), ('cat80', 'cat103'), ('cat73', 'cat82'), ('cat12', 'cat72'), ('cat80', 'cat12'), ('cat111', 'cat5'), ('cat2', 'cat111'), ('cat80', 'cat57'), ('cat80', 'cat79'), ('cat1', 'cat82'), ('cat11', 'cat13')]
            )],
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 13,
            'eta': 0.02,
            'colsample_bytree': 0.2,
            'subsample': 0.95,
            'gamma': 1.15,
            'alpha': 1.0,
        }, n_iter=8000, fair=1, transform_y=y_log_ofs(200), param_grid={'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]}),
    },

    'xgbf-ce-tst-2': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalAlphaEncoded(
                combinations=itertools.combinations('cat80,cat87,cat57,cat12,cat79,cat10,cat7,cat89,cat2,cat72,cat81,cat11,cat1,cat13,cat9,cat3,cat16,cat90,cat23,cat36,cat73,cat103,cat40,cat28,cat111,cat6,cat76,cat50,cat5,cat4,cat14,cat38,cat24,cat82,cat25'.split(','), 2)
            )],
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 13,
            'eta': 0.02,
            'colsample_bytree': 0.2,
            'subsample': 0.95,
            'gamma': 1.15,
            'alpha': 1.0,
        }, n_iter=2000, fair=1, transform_y=y_log_ofs(200), param_grid={'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]}),
    },


    'xgbf-ce-tst-3': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalAlphaEncoded(
                combinations=[('cat103', 'cat111'), ('cat2', 'cat6'), ('cat87', 'cat11'), ('cat103', 'cat4'), ('cat80', 'cat103'), ('cat73', 'cat82'), ('cat12', 'cat72'), ('cat80', 'cat12'), ('cat111', 'cat5'), ('cat2', 'cat111'), ('cat80', 'cat57'), ('cat80', 'cat79'), ('cat1', 'cat82'), ('cat11', 'cat13')]
            )],
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 13,
            'eta': 0.08,
            'colsample_bytree': 0.2,
            'subsample': 0.95,
            'gamma': 1.15,
            'alpha': 1.0,
        }, n_iter=2000, fair=1, transform_y=y_log_ofs(200), param_grid={'gamma': [0.9, 1.2], 'alpha': [0.8, 1.2], 'colsample_bytree': [0.15, 0.25]}),
    },

    'xgbf-ce-clrbf-1': {
        'features': ['numeric', 'categorical_encoded', 'cluster_rbf_200'],
        #'n_bags': 3,
        'model': Xgb({
            'max_depth': 8,
            'eta': 0.04,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'alpha': 0.9,
        }, n_iter=1250, fair=150, fair_decay=0.001),
    },

    'xgbf-ce-clrbf-2': {
        'features': ['numeric', 'categorical_encoded', 'cluster_rbf_50'],
        'n_bags': 3,
        'model': Xgb({
            'max_depth': 8,
            'eta': 0.04,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'alpha': 0.9,
            'lambda': 2.1
        }, n_iter=1100, fair=150),
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
        }, n_iter=1100, fair=1, transform_y=y_norm, param_grid={'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]}),
    },

    'xgbf-cm-tst': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalMeanEncoded(
                C=10000, noisy=True, noise_std=0.1, loo=False,
                combinations=itertools.combinations('cat80,cat87,cat57,cat12,cat79,cat10,cat7,cat89,cat2,cat72,cat81,cat11,cat1,cat13,cat9,cat3,cat16,cat90,cat23,cat36,cat73,cat103,cat40,cat28,cat111,cat6,cat76,cat50,cat5,cat4,cat14,cat38,cat24,cat82,cat25'.split(','), 2)
            )],
        'n_bags': 2,
        'model': Xgb({
            'max_depth': 7,
            'eta': 0.05,
            'colsample_bytree': 0.4,
            'subsample': 0.95,
            'min_child_weight': 4,
            'alpha': 0.0005,
        }, n_iter=500, fair=1, transform_y=y_norm, param_grid={'max_depth': [6, 7, 8], 'min_child_weight': [3, 4, 5]}),
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
        }, transform_y=y_norm),
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
        }, transform_y=y_norm),
    },

    'lgb-cd-1': {
        'features': ['numeric', 'categorical_dummy'],
        'n_bags': 4,
        'model': LightGBM({
            'num_iterations': 2150,
            'learning_rate': 0.01,
            'num_leaves': 200,
            'min_data_in_leaf': 8,
            'feature_fraction': 0.3,
            'bagging_fraction': 0.8,
            'bagging_freq': 20,
            'metric_freq': 10
        }, transform_y=y_norm),
    },

    'lgb-cd-2': {
        'features': ['numeric', 'categorical_dummy'],
        'n_bags': 4,
        'model': LightGBM({
            'num_iterations': 2900,
            'learning_rate': 0.01,
            'num_leaves': 200,
            'min_data_in_leaf': 8,
            'feature_fraction': 0.3,
            'bagging_fraction': 0.8,
            'bagging_freq': 20,
            'metric_freq': 10
        }, transform_y=y_log_ofs(200)),
    },

    'lgb-cd-tst': {
        'features': ['numeric', 'categorical_dummy'],
        'n_bags': 2,
        'model': LightGBM({
            'num_iterations': 5000,
            'learning_rate': 0.01,
            'num_leaves': 200,
            'min_data_in_leaf': 8,
            'feature_fraction': 0.3,
            'bagging_fraction': 0.8,
            'bagging_freq': 20,
            'metric_freq': 10,
            'metric': 'l1',
        }, transform_y=y_log_ofs(200)),
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
        }, transform_y=y_norm),
    },

    'libfm-cd': {
        'features': ['numeric_scaled', 'categorical_dummy'],
        'model': LibFM(params={
            'method': 'sgd',
            'learn_rate': 0.0001,
            'iter': 200,
            'dim': '1,1,12',
            'regular': '0,0,0.0002'
        }, transform_y=y_log),
    },

    'libfm-svd': {
        'features': ['svd'],
        'model': LibFM(params={
            'method': 'sgd',
            'learn_rate': 0.00007,
            'iter': 260,
            'dim': '1,1,12',
            'regular': '0,0,0.0002'
        }, transform_y=y_log),
    },

    'libfm-svd-clrbf': {
        'features': ['svd', 'cluster_rbf_50', 'cluster_rbf_100', 'cluster_rbf_200'],
        'model': LibFM(params={
            'method': 'sgd',
            'learn_rate': 0.00007,
            'iter': 350,
            'dim': '1,1,12',
            'regular': '0,0,0.0002'
        }, transform_y=y_log),
    },

    'nn-tst': {
        'features': ['numeric'],
        'model': Keras(nn_mlp, {'l1': 1e-3, 'l2': 1e-3, 'n_epoch': 1, 'batch_size': 128, 'layers': [10]}),
    },

    'nn1': {
        'features': ['numeric', 'categorical_encoded'],
        'model': Keras(nn_mlp, {'l1': 1e-3, 'l2': 1e-3, 'n_epoch': 100, 'batch_size': 48, 'layers': [400, 200], 'dropouts': [0.4, 0.2]}),
    },

    'nn-cd': {
        'features': ['numeric_scaled', 'categorical_dummy'],
        'n_bags': 2,
        'model': Keras(nn_mlp, lambda: {'l1': 1e-5, 'l2': 1e-5, 'n_epoch': 80, 'batch_size': 128, 'layers': [400, 200], 'dropouts': [0.4, 0.2], 'optimizer': Adadelta(), 'callbacks': [ExponentialMovingAverage(save_mv_ave_model=False)]}, scale=False),
    },

    'nn-cd-2': {
        'features': ['numeric_scaled', 'categorical_dummy'],
        'n_bags': 2,
        'model': Keras(nn_mlp, lambda: {'l1': 2e-5, 'l2': 2e-5, 'n_epoch': 40, 'batch_size': 128, 'layers': [400, 200, 100], 'dropouts': [0.5, 0.4, 0.2], 'batch_norm': True, 'optimizer': Adadelta(), 'callbacks': [ExponentialMovingAverage(save_mv_ave_model=False)]}, scale=False),
    },

    'nn-cd-3': {
        'features': ['numeric_scaled', 'categorical_dummy'],
        'n_bags': 4,
        'model': Keras(nn_mlp_2, lambda: {'n_epoch': 55, 'batch_size': 128, 'layers': [400, 200, 50], 'dropouts': [0.4, 0.25, 0.2], 'batch_norm': True, 'optimizer': Adadelta(), 'callbacks': [ExponentialMovingAverage(save_mv_ave_model=False)]}, scale=False, transform_y=y_log_ofs(200)),
    },

    'nn-cd-clrbf': {
        'features': ['numeric_scaled', 'categorical_dummy', 'cluster_rbf_200'],
        'n_bags': 4,
        'model': Keras(nn_mlp_2, lambda: {'n_epoch': 55, 'batch_size': 128, 'layers': [400, 200, 50], 'dropouts': [0.4, 0.25, 0.2], 'batch_norm': True, 'optimizer': Adadelta(), 'callbacks': [ExponentialMovingAverage(save_mv_ave_model=False)]}, scale=False, transform_y=y_log_ofs(200)),
    },

    'nn-cd-clrbf-2': {
        'features': ['numeric_scaled', 'categorical_dummy', 'cluster_rbf_50', 'cluster_rbf_100', 'cluster_rbf_200'],
        'n_bags': 4,
        'model': Keras(nn_mlp_2, lambda: {'n_epoch': 70, 'batch_size': 128, 'layers': [400, 200, 50], 'dropouts': [0.4, 0.25, 0.2], 'batch_norm': True, 'optimizer': Adadelta(), 'callbacks': [ExponentialMovingAverage(save_mv_ave_model=False)]}, scale=False, transform_y=y_log_ofs(200)),
    },

    'nn-cd-clrbf-tst': {
        'features': ['numeric_scaled', 'categorical_dummy', 'cluster_rbf_50', 'cluster_rbf_100', 'cluster_rbf_200'],
        'n_bags': 4,
        'model': Keras(nn_mlp_2, lambda: {'n_epoch': 70, 'batch_size': 128, 'layers': [300, 100], 'dropouts': [0.4, 0.25], 'batch_norm': True, 'optimizer': Adadelta(), 'callbacks': [ExponentialMovingAverage(save_mv_ave_model=False)]}, scale=False, transform_y=y_log_ofs(200)),
    },

    'nn-cd-tst': {
        'features': ['numeric_scaled', 'categorical_dummy'],
        'n_bags': 2,
        'model': Keras(nn_mlp_2, lambda: {'n_epoch': 100, 'batch_size': 128, 'layers': [400, 200, 50], 'dropouts': [0.4, 0.25, 0.2], 'batch_norm': True, 'optimizer': Adadelta(), 'callbacks': [ExponentialMovingAverage(save_mv_ave_model=False)]}, scale=False, transform_y=y_log_ofs(200)), # SGD(1e-3, momentum=0.9, nesterov=True, decay=1e-6)
    },

    'nn-cd-tst-2': {
        'features': ['numeric_scaled', 'numeric_edges', 'categorical_dummy'],
        'n_bags': 4,
        'model': Keras(nn_mlp_2, lambda: {'n_epoch': 100, 'batch_size': 128, 'layers': [400, 200, 50], 'dropouts': [0.4, 0.25, 0.2], 'batch_norm': True, 'optimizer': Adadelta(), 'callbacks': [ExponentialMovingAverage(save_mv_ave_model=False)]}, scale=False, transform_y=y_log_ofs(200)),
    },

    'nn-cd-lr': {
        'features': ['numeric_scaled', 'categorical_dummy'],
        #'n_bags': 2,
        'model': Keras(nn_lr, lambda: {'n_epoch': 3, 'batch_size': 128, 'layers': [], 'optimizer': Adadelta(), 'callbacks': [ExponentialMovingAverage(save_mv_ave_model=False)]}, scale=False, transform_y=y_log_ofs(200)),
    },

    'nn-svd': {
        'features': ['svd'],
        'n_bags': 2,
        'model': Keras(nn_mlp, lambda: {'l1': 1e-5, 'l2': 1e-5, 'n_epoch': 80, 'batch_size': 128, 'layers': [400, 200], 'dropouts': [0.4, 0.2], 'optimizer': Adadelta(), 'callbacks': [ExponentialMovingAverage(save_mv_ave_model=False)]}, scale=False),
    },

    'nn-svd-2': {
        'features': ['svd'],
        'n_bags': 2,
        'model': Keras(nn_mlp_2, lambda: {'l1': 1e-7, 'l2': 1e-7, 'n_epoch': 55, 'batch_size': 128, 'layers': [400, 200, 50], 'dropouts': [0.4, 0.2, 0.2], 'batch_norm': True, 'optimizer': Adadelta(), 'callbacks': [ExponentialMovingAverage(save_mv_ave_model=False)]}, scale=False, transform_y=y_log_ofs(200)),
    },

    'nn-cm-tst': {
        'features': ['numeric'],
        'feature_builders': [CategoricalMeanEncoded(1000)],
        'model': Keras(nn_mlp, lambda: {'l1': 1e-5, 'l2': 1e-5, 'n_epoch': 80, 'batch_size': 128, 'layers': [400, 200], 'dropouts': [0.4, 0.2], 'optimizer': Adadelta(), 'callbacks': [ExponentialMovingAverage(save_mv_ave_model=False)]}, scale=True),
    },

    'gb-ce': {
        'features': ['numeric', 'categorical_encoded'],
        'model': Sklearn(GradientBoostingRegressor(loss='lad', n_estimators=300, max_depth=7, max_features=0.2), param_grid={'n_estimators': (200, 400), 'max_depth': (6, 8), 'max_features': (0.1, 0.4)}),
    },

    'ab-ce': {
        'features': ['numeric', 'categorical_encoded'],
        'model': Sklearn(AdaBoostRegressor(loss='linear', n_estimators=300), transform_y=y_log_ofs(200), param_grid={'n_estimators': (50, 400), 'learning_rate': (0.1, 1.0)}),
    },

    'et-ce': {
        'features': ['numeric', 'categorical_encoded'],
        'model': Sklearn(ExtraTreesRegressor(50, max_features=0.2, n_jobs=-1), transform_y=y_log),
    },

    'et-ce-2': {
        'features': ['numeric', 'categorical_encoded'],
        'model': Sklearn(ExtraTreesRegressor(50, max_features=0.2, n_jobs=-1), transform_y=y_log_ofs(200)),
    },

    'et-ce-3': {
        'features': ['numeric', 'categorical_encoded'],
        'model': Sklearn(ExtraTreesRegressor(50, max_features=0.8, min_samples_split=26, max_depth=23, n_jobs=-1), transform_y=y_log_ofs(200)),
    },

    'et-ce-4': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalAlphaEncoded(
                combinations=[('cat103', 'cat111'), ('cat2', 'cat6'), ('cat87', 'cat11'), ('cat103', 'cat4'), ('cat80', 'cat103'), ('cat73', 'cat82'), ('cat12', 'cat72'), ('cat80', 'cat12'), ('cat111', 'cat5'), ('cat2', 'cat111'), ('cat80', 'cat57'), ('cat80', 'cat79'), ('cat1', 'cat82'), ('cat11', 'cat13')]
            )],
        'model': Sklearn(ExtraTreesRegressor(400, max_features=0.623,  max_depth=29, min_samples_leaf=4, n_jobs=-1), transform_y=y_log_ofs(200), param_grid={'min_samples_leaf': (2, 40), 'max_features': (0.05, 0.95), 'max_depth': (5, 40)}),
    },

    'rf-ce': {
        'features': ['numeric', 'categorical_encoded'],
        'model': Sklearn(RandomForestRegressor(100, max_features=0.2, n_jobs=-1), transform_y=y_log),
    },

    'rf-ce-2': {
        'features': ['numeric', 'categorical_encoded'],
        'model': Sklearn(RandomForestRegressor(100, min_samples_split=16, max_features=0.3, max_depth=26, n_jobs=-1), transform_y=y_log_ofs(200)),
    },

    'rf-ce-3': {
        'features': ['numeric'],
        'feature_builders': [
            CategoricalAlphaEncoded(
                combinations=[('cat103', 'cat111'), ('cat2', 'cat6'), ('cat87', 'cat11'), ('cat103', 'cat4'), ('cat80', 'cat103'), ('cat73', 'cat82'), ('cat12', 'cat72'), ('cat80', 'cat12'), ('cat111', 'cat5'), ('cat2', 'cat111'), ('cat80', 'cat57'), ('cat80', 'cat79'), ('cat1', 'cat82'), ('cat11', 'cat13')]
            )],
        'model': Sklearn(RandomForestRegressor(400, max_features=0.62, max_depth=39, min_samples_leaf=5, n_jobs=-1), transform_y=y_log_ofs(200), param_grid={'min_samples_leaf': (2, 40), 'max_features': (0.05, 0.95), 'max_depth': (5, 40)}),
    },

    'lr-cd': {
        'features': ['numeric_scaled', 'categorical_dummy'],
        'model': Sklearn(Ridge(1e-3), transform_y=y_log, param_grid={'C': (1e-3, 1e3)}),
    },

    'lr-cm': {
        'features': ['numeric_scaled'],
        'feature_builders': [
            CategoricalMeanEncoded(
                C=10000, noisy=True, noise_std=0.01, loo=True,
                combinations=itertools.combinations('cat80,cat87,cat57,cat12,cat79,cat10,cat7,cat89,cat2,cat72,cat81,cat11,cat1,cat13,cat9,cat3,cat16,cat90,cat23,cat36,cat73,cat103,cat40,cat28,cat111,cat6,cat76,cat50,cat5,cat4,cat14,cat38,cat24,cat82,cat25'.split(','), 2)
            )],
        'model': Sklearn(Ridge(1e-3), transform_y=y_log, param_grid={'C': (1e-3, 1e3)}),
    },

    'lr-cd-nr': {
        'features': ['numeric_rank_norm', 'categorical_dummy'],
        'model': Sklearn(Ridge(1e-3), transform_y=y_log, param_grid={'C': (1e-3, 1e3)}),
    },

    'lr-ce': {
        'features': ['numeric', 'categorical_encoded'],
        'model': Sklearn(Pipeline([('sc', StandardScaler(with_mean=False)), ('lr', Ridge(1e-3))]), transform_y=y_log),
    },

    'lr-svd': {
        'features': ['svd'],
        'model': Sklearn(Ridge(1e-3), transform_y=y_log),
    },

    'lr-svd-clrbf': {
        'features': ['svd', 'cluster_rbf_200'],
        'model': Sklearn(Ridge(1e-3), transform_y=y_log_ofs(200)),
    },

    'lr-svd-clrbf-2': {
        'features': ['svd', 'cluster_rbf_50', 'cluster_rbf_100', 'cluster_rbf_200'],
        'model': Sklearn(Ridge(1e-3), transform_y=y_log_ofs(200)),
    },

    'knn1': {
        'features': ['svd'],
        'model': Sklearn(Pipeline([('sc', StandardScaler(with_mean=False)), ('knn', KNeighborsRegressor(5))]), transform_y=y_log),
    },

    'lsh1': {
        'features': ['numeric', 'categorical_encoded'],
        'model': LshForest(transform_y=y_log),
    },

    'l2-nn': {
        'predictions': l1_predictions,
        'n_bags': 4,
        'model': Keras(nn_mlp, lambda: {'l1': 1e-5, 'l2': 1e-5, 'n_epoch': 30, 'batch_size': 128, 'layers': [50], 'dropouts': [0.1], 'optimizer': SGD(1e-3, momentum=0.8, nesterov=True, decay=3e-5), 'callbacks': [ReduceLROnPlateau(patience=5, factor=0.2, cooldown=3), ExponentialMovingAverage(save_mv_ave_model=False)]}),
    },

    'l2-nn-2': {
        'predictions': l1_predictions,
        'n_bags': 4,
        'model': Keras(nn_mlp, lambda: {'l1': 1e-5, 'l2': 1e-5, 'n_epoch': 30, 'batch_size': 128, 'layers': [200, 50], 'dropouts': [0.15, 0.1], 'optimizer': SGD(1e-3, momentum=0.8, nesterov=True, decay=3e-5), 'callbacks': [ReduceLROnPlateau(patience=5, factor=0.2, cooldown=3)]}),
    },

    'l2-lr': {
        'predictions': l1_predictions,
        'prediction_transform': np.log,
        'model': Sklearn(Ridge(), transform_y=y_log),
    },

    'l2-xgbf': {
        'predictions': l1_predictions,
        'n_bags': 4,
        'model': Xgb({
            'max_depth': 4,
            'eta': 0.01,
            'colsample_bytree': 0.3,
            'subsample': 0.75,
            'min_child_weight': 6,
        }, n_iter=2200, fair=1.0, transform_y=y_log_ofs(200)),
    },

    'l2-et': {
        'predictions': l1_predictions,
        'model': Sklearn(ExtraTreesRegressor(100, max_depth=11, max_features=0.8, n_jobs=-1), transform_y=y_log_ofs(200), param_grid={'min_samples_leaf': (1, 40), 'max_features': (0.05, 0.8), 'max_depth': (3, 20)}),
    },

    'l2-rf': {
        'predictions': l1_predictions,
        'model': Sklearn(RandomForestRegressor(100, max_depth=9, max_features=0.8, min_samples_leaf=23, n_jobs=-1), transform_y=y_log_ofs(200), param_grid={'min_samples_leaf': (1, 40), 'max_features': (0.05, 0.8), 'max_depth': (3, 20)}),
    },

    'l3-nn': {
        'predictions': l2_predictions,
        'n_bags': 4,
        'model': Keras(nn_lr, lambda: {'l2': 1e-5, 'n_epoch': 1, 'batch_size': 128, 'optimizer': SGD(lr=2.0, momentum=0.8, nesterov=True, decay=1e-4)}),
        'agg': np.mean,
    },
}

print "Preset: %s" % args.preset

preset = presets[args.preset]

feature_builders = preset.get('feature_builders', [])

n_bags = preset.get('n_bags', 1)

y_aggregator = preset.get('agg', np.mean)

print "Loading train data..."
train_x = load_x('train', preset)
train_y = Dataset.load_part('train', 'loss')
train_p = pd.Series(0.0, index=Dataset.load_part('train', 'id'))
train_r = Dataset.load('train', parts=np.unique(sum([b.requirements for b in feature_builders], ['loss'])))

feature_names = extract_feature_names(preset)


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
test_p = np.zeros((test_x.shape[0], n_bags * n_folds))
test_r = Dataset.load('test', parts=np.unique([b.requirements for b in feature_builders]))

maes = []

for fold, (fold_train_idx, fold_eval_idx) in enumerate(KFold(len(train_y), n_folds, shuffle=True, random_state=2016)):
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

    fold_feature_names = list(feature_names)

    if len(feature_builders) > 0:  # TODO: Move inside of bagging loop
        print "    Building per-fold features..."

        fold_train_x = [fold_train_x]
        fold_eval_x = [fold_eval_x]
        fold_test_x = [fold_test_x]

        for fb in feature_builders:
            fold_train_x.append(fb.fit_transform(fold_train_r))
            fold_eval_x.append(fb.transform(fold_eval_r))
            fold_test_x.append(fb.transform(fold_test_r))
            fold_feature_names += fb.get_feature_names()

        fold_train_x = hstack(fold_train_x)
        fold_eval_x = hstack(fold_eval_x)
        fold_test_x = hstack(fold_test_x)

    eval_p = np.zeros((fold_eval_x.shape[0], n_bags))

    for bag in xrange(n_bags):
        print "    Training model %d..." % bag

        pe, pt = preset['model'].fit_predict(train=(fold_train_x, fold_train_y),
                                             val=(fold_eval_x, fold_eval_y),
                                             test=(fold_test_x, ),
                                             seed=42 + 13*bag,
                                             feature_names=fold_feature_names)

        eval_p[:, bag] += pe
        test_p[:, fold * n_bags + bag] += pt

    print "  MAE of mean: %.5f" % mean_absolute_error(fold_eval_y, np.mean(eval_p, axis=1))
    print "  MAE of median: %.5f" % mean_absolute_error(fold_eval_y, np.median(eval_p, axis=1))

    # Aggregate eval predictions
    eval_p = y_aggregator(eval_p, axis=1)

    # Normalize train predictions
    train_p.iloc[fold_eval_idx] += eval_p

    # Calculate err
    maes.append(mean_absolute_error(fold_eval_y, eval_p))

    print "  MAE: %.5f" % maes[-1]

    # Free mem
    del fold_train_x, fold_train_y, fold_eval_x, fold_eval_y


## Analyzing predictions

test_p = pd.Series(y_aggregator(test_p, axis=1), index=Dataset.load_part('test', 'id'))

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
