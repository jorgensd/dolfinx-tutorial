# The equations of linear elasticity

Authors: Anders Logg and Hans Petter Langtangen

Analysis of structures is one of the major activities of modern engineering, which likely makes the PDE modelling the deformation of elastic bodies the most popular PDE in the world. It takes just one page of code to solve the equations of 2D or 3D elasticity in DOLFINx, and shown in this section.

## The PDE problem
The equations governing small elastic deformations of a body $\Omega$ can be written as
```{math}
:label: elasticity-PDE
    -\nabla \cdot \sigma (u) &= f && \text{in } \Omega\\
    \sigma(u)&= \lambda \mathrm{tr}(\epsilon(u))I + 2 \mu \epsilon(u)\\
    \epsilon(u) &= \frac{1}{2}\left(\nabla u + (\nabla u )^T\right)
```
where $\sigma$ is the stress tensor, $f$ is the body force per unit volume, $\lambda$ and $\mu$ are Lamé's elasticity parameters for the material in $\Omega$, $I$ is the identity tensor, $\mathrm{tr}$ is the trace operator on a tensor, $\epsilon$ is the symmetric strain tensor (symmetric gradient), and $u$ is the displacement vector field. Above we have assumed isotropic elastic conditions.
By inserting $\epsilon(u)$ into $\sigma$ we obtain 
\begin{align}
    \sigma(u)&=\lambda(\nabla \cdot u)I + \mu(\nabla u + (\nabla u)^T)
\end{align}
Note that we could have written the PDE above as a single vector PDE for $u$, which is the governing PDE for the unknown $u$ (Navier's) equation. However, it is convenient to keep the current representation of the PDE for the derivation of the variational formulation.

## The variational formulation
The variational formulation of the PDE consists of forming the inner product of the PDE [](elasticity-PDE) with a *vector* test function $v\in\hat{V}$, where $\hat{V}$ is a vector-valued test function space, and integrating over the domain $\Omega$:
```{math}
    -\int_{\Omega}(\nabla \cdot \sigma)\cdot v ~\mathrm{d} x = \int_{\Omega} f\cdot v ~\mathrm{d} x.
```
Since $\nabla \cdot \sigma$ contains second-order derivatives of our unknown $u$, we integrate this term by parts
```{math}
    -\int_{\Omega}(\nabla \cdot \sigma)\cdot v ~\mathrm{d} x =\int_{\Omega}\sigma : \nabla v ~\mathrm{d}x - \int_{\partial\Omega} (\sigma \cdot n)\cdot v~\mathrm{d}x,
```
where the colon operator is the inner product between tensors (summed pairwise product of all elements), and $n$ is the outward unit normal at the boundary. The quantity $\sigma \cdot n$ is known as the *traction* or stress vector at the boundary, and often prescribed as a boundary condition. We here assume that it is prescribed on a part $\partial \Omega_T$ of the boundary as $\sigma \cdot n=T$. On the remaining part of the boundary, we assume that the value of the displacement is given as Dirichlet condition (and hence the boundary integral on those boundaries are $0$). We thus obtain
```{math}
    \int_{\Omega} \sigma : \nabla v ~\mathrm{d} x = \int_{\Omega} f\cdot v ~\mathrm{d} x + \int_{\partial\Omega_T}Tv~\mathrm{d} s.
```
If we now insert for $\sigma$ its representation with the unknown $u$, we can obtain our variational formulation:
Find $u\in V$ such that 
```{math}
    a(u,v) = L(v)\qquad  \forall v \in \hat{V},
```
where
```{math}
:label: elasticity
    a(u,v)&=\int_{\Omega}\sigma(u):\nabla v ~\mathrm{d}x\\
    \sigma(u)&=\lambda(\nabla \cdot u)I+\mu (\nabla u + (\nabla u)^T),\\
    L(v)&=\int_{\Omega}f\cdot v~\mathrm{d} x + \int_{\partial\Omega_T}T\cdot v~\mathrm{d}s.
```
One can show that the inner product of a symmetric tensor $A$ and an anti-symmetric tensor $B$ vanishes. If we express $\nabla v$ as a sum of its symmetric and anti-symmetric parts, only the symmetric part will survive in the product $\sigma : \nabla v$ since $\sigma$ is a symmetric tensor. Thus replacing $\nabla v$ by the symmetric gradient $\epsilon(v)$ gives rise to a slightly different variational form
```{math}
:label: elasticity-alternative
    a(u,v)= \int_{\Omega}\sigma(u):\epsilon(v)~\mathrm{d} x,
```
where $\epsilon(v)$ is the symmetric part of $\nabla v$:
```{math}
    \epsilon(v)=\frac{1}{2}\left(\nabla v + (\nabla v)^T\right)
```
The formulation [](elasticity-alternative) is what naturally arises from minimization of elastic potential energy and is a more popular formulation than [](elasticity).
