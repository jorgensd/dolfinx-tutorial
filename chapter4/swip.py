# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.2
# ---

# # Symmetric weighted interior penalty method for advection diffusion equations.
#
# Author: Jørgen S. Dokken

# The Symmetric Weighted Interior Penalty (SWIP) method is a numerical method
# for solving advection-diffusion equations. It was first introduced in {cite}`ern2008swip`.
# The following code implements the SWIP method for the analytical solution presetened in ....

# The strong formulaton of the advection-diffusion equation is given by
#
# \begin{align}
# - \nabla \cdot (K \nabla c) + \mathbf{u} \cdot \nabla c + \mu c &= f && \text{in } \Omega, \\
# c &= g && \text{on } \partial \Omega, \\
# \end{align}
#
# where $K$ is the symmetric, positive definite diffusivity tensor, $\mathbf{u}$ is the advection velocity.
# For restrictions on $K$ and $\mathbf{u}$, see {cite}`ern2008swip.
#
# ```{admonition} Continuity of $K$
# A key assumption that we cannot skip is that $K$ is continous inside each element,
# but can be discontinuous across element boundaries.
# ```

# There are two values that can be tuned for the SWIP method:
# - The positive penalty parameter $\gamma$.
# - A weighting parameter $\omega$, which over an interior facet is defined such
#   that $\omega^+ + \omega^- = 1$, where $^+$ and $^-$ denote the restriction of $\Omega to each
#   side of the facet.

# We write the weak formulation as:
# Find $c_h \in V_h$ such that
#
# \begin{align}
# a(c_h, v_h) &= (f, v_h)_\Omega && \forall v_h \in V_h,
# \end{align}
#
# where $(\cdot, \cdot)_\Omega$ is the $L^2$ inner product over $\Omega$ and

# \begin{align}
# a(c, v) &= (K\nabla c, \nabla v)_\Omega + (\mu c, v)_\Omega + (\mathbf{u} \cdot \nabla c, v)_\Omega \\
# &+= \int_{\partial \Omega} -K\nabla c \cdot \mathbf{n} v - K \nabla v \cdot \mathbf{n} c + \gamma c v ~\mathrm{d}s\\
# &+ \sum_{F\in \mathcal{F}_{h^i}} (\beta \cdot \mathbf{n}^+\{w\} \left[ v \right] )_F
# \end{align}
#
# where we recognize the boundary terms from [the Nitsche method](../chapter1/nitsche)

from mpi4py import MPI
