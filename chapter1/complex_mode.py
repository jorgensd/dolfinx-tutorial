# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (DOLFINx complex)
#     language: python
#     name: python3-complex
# ---

# # Running DOLFINx in complex mode
#
# Author: Jørgen S. Dokken
#
# This section will explain how to run DOLFINx in complex mode, and its peculiarities.
#
# ## TODO: Add complex example

import dolfinx
from petsc4py import PETSc
import numpy as np
assert np.dtype(PETSc.ScalarType).kind == 'c'


