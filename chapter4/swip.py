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
# For restrictions on $K$ and $\mathbf{u}$, see {cite}`ern2008swip`.
#
# ```{admonition} Continuity of $K$
# A key assumption that we cannot skip is that $K$ is continous inside each element,
# but can be discontinuous across element boundaries.
# ```

# There are two values that can be tuned for the SWIP method:
# - The positive penalty parameter $\gamma$.
# - A weighting parameter $\omega$, which over an interior facet is defined such
#   that $\omega^+ + \omega^- = 1$, where $^+$ and $^-$ denote the restriction of $\Omega$ to each
#   side of the facet. Specifically, we choose $\omega^- = \frac{\delta_{K_n}^+}{\delta_{K_n}^- + \delta_{K_n}^+}$ and
#   $\omega^+ = \frac{\delta_{K_n}^-}{\delta_{K_n}^- + \delta_{K_n}^+}$, where
#   $\delta_{K_n}^\mp=n^T_FK^\mp n_F$.
#
# The set of interior facets $\mathcal{F}_{h^i}$ is defined as the set of facets that are connected to two cells, $\{T^+, T^-\}$.
# We defined the average as such a facet as $\{v\}_F=\frac{1}{2}(v^{+}+v^{-})$, the weighted average
# as $\{v\}_\omega = \omega^+ v^+ + \omega^- v^-$, and the jump as $[v] = v^+ - v^-$ and
# $\mathbf{n}_F$ is a normal pointing out of $T^-$ towards $T^+$.

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
# & -(K\nabla c\cdot\mathbf{n}, v)_{\partial\Omega}
# - (K \nabla v \cdot \mathbf{n}, c)_{\partial\Omega}
# + (\gamma c, v)_{\partial\Omega}
# -\frac{1}{2}(\mathbf{u}\cdot \mathbf{n} c, v)_{\partial\Omega}\\
# &+ \sum_{F\in \mathcal{F}_{h^i}}
# \Big(
# -(\{K\nabla c\}_\omega\cdot\mathbf{n}_F, [ v ])_{F}
# -(\{K\nabla v\}_\omega\cdot\mathbf{n}_F, [ c ])_{F}
# + (\gamma [c], [v])_F\\
# &\hspace{1ex}\qquad\qquad-(\mathbf{u} \cdot \mathbf{n}_F\{v\}, \left[ c \right] )_F
# \Big).
# \end{align}
#
# where the penalty parameter $\gamma$ is defined as
#
# \begin{align}
# \gamma = \alpha \frac{\gamma_K}{h_f}+\gamma_\mathbf{u}
# \end{align}
#
# where $\alpha>0$ can vary from face to face and
#
# \begin{align}
# \gamma_K &=
# \begin{cases}
# \frac{\delta_{K_n}^+\delta_{K_n}^-}{\delta_{K_n}^- + \delta_{K_n}^+} &\text{if } F\in \mathcal{F}_{h^i}\\
# \delta_{K_n} &\text{if } F\in \partial\Omega
# \end{cases}\\
# \gamma_\mathbf{u} &= \frac{1}{2}\vert \mathbf{u} \cdot \mathbf{n}_F \vert ~\quad\qquad\qquad\qquad\text{if } F \in \mathcal{F}_{h^i}\\
# \end{align}
#
# where we recognize the first three boundary terms from [the Nitsche method](../chapter1/nitsche)

from mpi4py import MPI


# ```{bibliography}
#    :filter: cited and ({"chapter4/swip"} >= docnames)
# ```
