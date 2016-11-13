import numpy as np

from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils.fixes import parallel_helper
from sklearn.ensemble.base import _partition_estimators

from sklearn.ensemble import ExtraTreesRegressor


class MedianExtraTreesRegressor(ExtraTreesRegressor):

    def predict(self, X):
        """Predict regression target for X.
        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest.
        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.
        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted values.
        """

        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # Parallel loop
        all_y_hat = Parallel(n_jobs=n_jobs, verbose=self.verbose,
                             backend="threading")(
            delayed(parallel_helper)(e, 'predict', X, check_input=False)
            for e in self.estimators_)

        return np.median(all_y_hat, axis=0)
