import pytest

from sklearn.utils.estimator_checks import check_estimator

from gibbs_sampler import GibbsSampler


@pytest.mark.parametrize(
    "estimator",
    [GibbsSampler()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
