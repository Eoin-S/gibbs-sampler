import pytest
import numpy as np

from sklearn.datasets import load_iris
from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose

from gibbs_sampler import GibbsSampler


@pytest.fixture
def data():
    return load_iris(return_X_y=True)

def test_template_estimator(data):
    est = GibbsSampler()
    assert est.slope == 0

    est.fit(*data)
    assert hasattr(est, 'is_fitted_')

    X = data[0]
    y_pred = est.predict(X)
    assert_array_equal(y_pred, np.ones(X.shape[0], dtype=np.int64))


