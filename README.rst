.. -*- mode: rst -*-

|Travis|_ |AppVeyor|_ |Codecov|_ |CircleCI|_ |ReadTheDocs|_


.. |Travis| image:: https://api.travis-ci.com/Eoin-S/gibbs-sampler.svg?branch=master
.. _Travis: https://app.travis-ci.com/github/Eoin-S/gibbs-sampler

.. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/coy2qqaqr1rnnt5y/branch/master?svg=true
.. _AppVeyor: https://ci.appveyor.com/project/Eoin-S/gibbs-sampler

.. |Codecov| image:: https://codecov.io/gh/scikit-learn-contrib/project-template/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/scikit-learn-contrib/project-template

.. |CircleCI| image:: https://circleci.com/gh/scikit-learn-contrib/project-template.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://app.circleci.com/pipelines/github/Eoin-S/gibbs-sampler

.. |ReadTheDocs| image:: https://readthedocs.org/projects/gibbs-sampling/badge/?version=latest
.. _ReadTheDocs: https://gibbs-sampling.readthedocs.io/en/latest/?badge=latest

Gibbs Sampling
============================================================

.. _scikit-learn: https://scikit-learn.org

The **Gibbs Sampler** or otherwise known **Gibbs Sampling** is a computationally convenient Bayesian inference algorithm. This package implements the Gibbs Sampler using the APIs of scikit-learn objects, which enables the estimator to safely interact with scikit-learn Pipelines and model selection tools.

Installation
============================================================

PyPi: python -m pip install tslearn

Contributing
============================================================

If you would like to contribute a feature then fork the master branch (*fork the release if you are fixing a bug*). Be sure to run the tests before changing any code. The following command will run all the tests:

  python setup.py test

Let us know what you want to do just in case we're already working on an implementation of something similar. This way we can avoid any needless duplication of effort. Also, please don't forget to add tests for any new functions.
