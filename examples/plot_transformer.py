"""
=============================
Plotting Template Transformer
=============================

An example plot of :class:`skltemplate.template.TemplateTransformer`
"""
import numpy as np
from matplotlib import pyplot as plt
from gibbs_sampler import GibbsSampler

estimator = GibbsSampler()

print(estimator.get_params())

