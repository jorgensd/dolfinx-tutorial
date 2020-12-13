# A nonlinear Poisson equation
Authors: Anders Logg and Hans Petter Langtangen

We shall now adress how to solve nonlinear PDEs. We will see that non-linear problems introduce some subtle differences on how we define the variational form.

## The PDE problem
As a model for the solution of nonlinear PDEs, we take the following non-linear Poisson equation
\begin{align}
    - \nabla \cdot (q(u) \nabla u)&=f && \text{in } \Omega,\\
    u&=u_D && \text{on } \partial \Omega,
\end{align}
and the coefficients $q(u)$ makes the problem non-linear (unless q(u) is constant in $u$).

## Variational  formulation
As usual, we multiply the PDE by a test function $v\in \hat{V}$, integrate over the domain, and integrate second-order derivatives byparts. The boundary integrals arisng from integration by parts vanishes wherever we employ Dirichlet conditions. The resulting variational formulation of our model problem becomes:

Find $u\in V$ such that
\begin{align}
    F(u; v)&=0 && \forall v \in \hat{V},
\end{align}
where
\begin{align}
    F(u; v)&=\int_{\Omega}(q(u)\nabla u \cdot \nabla v - fv)\mathrm{d}x,
\end{align}
and 
\begin{align}
    V&=\left\{v\in H^1(\Omega)\vert v=u_D \text{on } \partial \Omega \right\}\\
    \hat{V}&=\left\{v\in H^1(\Omega)\vert v=0 \text{on } \partial \Omega \right\}
\end{align}

The discrete problem arises as usual by restricting $V$ and $\hat{V}$ to a pair of discrete spaces. The discrete non-linear problem can therefore be written as:

Find $u_h \in V_h$ such that
\begin{align}
F(u_h, v) &=0 \quad \forall v \in \hat{V}_h,
\end{align}
with $u_h=\sum_{j=1}^N U_j\phi_j$. Since $F$ is non-linear in $u$, the variational statement gives rise to a system of non-linear algebraic equation in  the unknowns $U_1,\dots,U_N$.