# This is a callback function to be used with training of Keras models.
# It create an exponential moving average of a model (trainable) weights.
# This functionlity is already available in TensorFlow:
# https://www.tensorflow.org/versions/r0.10/api_docs/python/train.html#ExponentialMovingAverage
# and can often be used to get better validation/test performance. For an
# intuitive explantion on why to use this, see 'Model Ensembles" section here:
# http://cs231n.github.io/neural-networks-3/

import numpy as np
import scipy.sparse as sp

from keras import backend as K
from keras.callbacks import Callback
from keras.models import load_model
from keras.engine.training import collect_trainable_weights

import sys
import warnings


class ExponentialMovingAverage(Callback):
    """create a copy of trainable weights which gets updated at every
       batch using exponential weight decay. The moving average weights along
       with the other states of original model(except original model trainable
       weights) will be saved at every epoch if save_mv_ave_model is True.
       If both save_mv_ave_model and save_best_only are True, the latest
       best moving average model according to the quantity monitored
       will not be overwritten. Of course, save_best_only can be True
       only if there is a validation set.
       This is equivalent to save_best_only mode of ModelCheckpoint
       callback with similar code. custom_objects is a dictionary
       holding name and Class implementation for custom layers.
       At end of every batch, the update is as follows:
       mv_weight -= (1 - decay) * (mv_weight - weight)
       where weight and mv_weight is the ordinal model weight and the moving
       averaged weight respectively. At the end of the training, the moving
       averaged weights are transferred to the original model.
       """
    def __init__(self, decay=0.999, filepath='temp_weight.hdf5',
                 save_mv_ave_model=True, verbose=0,
                 save_best_only=False, monitor='val_loss', mode='auto',
                 save_weights_only=False, custom_objects={}):
        self.decay = decay
        self.filepath = filepath
        self.verbose = verbose
        self.save_mv_ave_model = save_mv_ave_model
        self.save_weights_only = save_weights_only
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.custom_objects = custom_objects  # dictionary of custom layers
        self.sym_trainable_weights = None  # trainable weights of model
        self.mv_trainable_weights_vals = None  # moving averaged values
        super(ExponentialMovingAverage, self).__init__()

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_train_begin(self, logs={}):
        self.sym_trainable_weights = collect_trainable_weights(self.model)
        # Initialize moving averaged weights using original model values
        self.mv_trainable_weights_vals = {x.name: K.get_value(x) for x in
                                          self.sym_trainable_weights}
        if self.verbose:
            print('Created a copy of model weights to initialize moving'
                  ' averaged weights.')

    def on_batch_end(self, batch, logs={}):
        for weight in self.sym_trainable_weights:
            old_val = self.mv_trainable_weights_vals[weight.name]
            self.mv_trainable_weights_vals[weight.name] -= \
                (1.0 - self.decay) * (old_val - K.get_value(weight))

    def on_epoch_end(self, epoch, logs={}):
        """After each epoch, we can optionally save the moving averaged model,
        but the weights will NOT be transferred to the original model. This
        happens only at the end of training. We also need to transfer state of
        original model to model2 as model2 only gets updated trainable weight
        at end of each batch and non-trainable weights are not transferred
        (for example mean and var for batch normalization layers)."""
        if self.save_mv_ave_model:
            filepath = self.filepath.format(epoch=epoch, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best moving averaged model only '
                                  'with %s available, skipping.'
                                  % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('saving moving average model to %s'
                                  % (filepath))
                        self.best = current
                        model2 = self._make_mv_model(filepath)
                        if self.save_weights_only:
                            model2.save_weights(filepath, overwrite=True)
                        else:
                            model2.save(filepath, overwrite=True)
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving moving average model to %s' % (epoch, filepath))
                model2 = self._make_mv_model(filepath)
                if self.save_weights_only:
                    model2.save_weights(filepath, overwrite=True)
                else:
                    model2.save(filepath, overwrite=True)

    def on_train_end(self, logs={}):
        for weight in self.sym_trainable_weights:
            K.set_value(weight, self.mv_trainable_weights_vals[weight.name])

    def _make_mv_model(self, filepath):
        """ Create a model with moving averaged weights. Other variables are
        the same as original mode. We first save original model to save its
        state. Then copy moving averaged weights over."""
        self.model.save(filepath, overwrite=True)
        model2 = load_model(filepath, custom_objects=self.custom_objects)

        for w2, w in zip(collect_trainable_weights(model2), collect_trainable_weights(self.model)):
            K.set_value(w2, self.mv_trainable_weights_vals[w.name])

        return model2


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
