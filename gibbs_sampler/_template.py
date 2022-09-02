"""
This is a module to be used as a reference for building other modules
"""
from abc import ABCMeta, abstractmethod
#from sklearn.base import LinearModel
import numpy as np
from sklearn import linear_model
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

class GibbsSampler(BaseEstimator ,RegressorMixin):
    """ A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    slope : int, default=0
        Initial guess on slope, independent variable.
    intercept : int, default=0
        Initial guess on intercept, independent variable.
    gamma : float, default=2.0
        gamma defines the precision of the target model
    n_iter : int, default=2.0
        Number of iterations excuted

    Examples
    --------
    >>> from gibbs_sampler import GibbsSampler
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.zeros((100, ))
    >>> estimator = GibbsSampler()
    >>> estimator.fit(X, y)
    GibbsSampler()
    """
    def __init__(
            self, 
            slope=0, 
            intercept=0, 
            gamma=2, 
            n_iter=1000, 
            mu_0=0, 
            tau_0=1, 
            mu_1=0, 
            tau_1=1, 
            alpha=2, 
            beta=1,
            random_state=None,
            ):
        self.n_iter = n_iter
        self.intercept = intercept
        self.slope = slope
        self.gamma = gamma
        self.mu_0 = mu_0
        self.tau_0 = tau_0
        self.mu_1 = mu_1
        self.tau_1 = tau_1
        self.alpha = alpha
        self.beta = beta
        self.random_state = random_state

    def fit(self, X, y):
        """A reference implementation of a fitting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, accept_sparse=True)
        self.is_fitted_ = True
        
        trace = np.zeros((self.n_iter, 3)) ## trace to store values of beta_0, beta_1, tau
        
        for i in range(self.n_iter):
            
            c = sample_beta_0(y, X, self.slope, self.gamma, self.mu_0, self.tau_0)
            m = sample_beta_1(y, X, self.intercept, self.gamma, self.mu_1, self.tau_1)
            
            tau = sample_tau(y, X, c, m, self.alpha, self.beta)
            trace[i,:] = np.array((c, m, tau))
        
        self.coef_ = np.mean(trace[0,:]), np.mean(trace[1,:])
        
        # `fit` should always return `self`
        return self

    def predict(self, X):
        """ A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        
        return self.coef_[1] + self.coef_[0] * X


def sample_beta_0(y, x, beta_1, tau, mu_0, tau_0):
    N = len(y)
    assert len(x) == N
    precision = tau_0 + tau * N
    mean = tau_0 * mu_0 + tau * np.sum(y - beta_1 * x)
    mean /= precision
    return np.random.normal(mean, 1 / np.sqrt(precision))

def sample_beta_1(y, x, beta_0, tau, mu_1, tau_1):
    N = len(y)
    assert len(x) == N
    precision = tau_1 + tau * np.sum(x * x)
    mean = tau_1 * mu_1 + tau * np.sum( (y - beta_0) * x)
    mean /= precision
    return np.random.normal(mean, 1 / np.sqrt(precision))

def sample_tau(y, x, beta_0, beta_1, alpha, beta):
    N = len(y)
    alpha_new = alpha + N / 2
    resid = y - beta_0 - beta_1 * x
    beta_new = beta + np.sum(resid * resid) / 2
    return np.random.gamma(alpha_new, 1 / beta_new)
