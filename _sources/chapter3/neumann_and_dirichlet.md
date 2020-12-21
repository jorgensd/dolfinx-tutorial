# Subdomains and boundary conditions
Authors: Anders Logg and Hans Petter Langtangen

So far, we have only looked briefly at how to specify boundary conditions.
In this chapter, we look more closely at how to specify boundary conditions on specific (subdomains) of the boundary and how to combine multiple boundary conditions.
We will also look at how to generate meshes with subdomains and how to define coefficients with different values in different
subdomains.

## Combining Dirichlet and Neumann conditions

Let's return to the Poisson problem from the [Fundamentals chapter](./../chapter1/fundamentals.md) and see how to extend the mathematics and the implementation to handle Dirichlet condition in combination with a Neumann condition.
The domain is still the unit square, but now we set the Dirichlet condition $u=u_D$ at the left and right sides, while the Neumann condition 
```{math}
-\frac{\partial u}{\partial n}=g
```
is applied to the  remaining sides $y=0$ and $y=1$.

## The PDE problem
Let $\Lambda_D$ and $\Lambda_N$ denote parts of the boundary $\partial \Omega$ where the Dirichlet and Neumann conditions apply, respectively.
The complete boundary-value problem can be written as
```{math}
-\nabla^2 u &=f && \text{in } \Omega,\\
u&=u_D &&\text{on } \Lambda_D.\\
-\frac{\partial u}{\partial n}&=g && \text{on }\Lambda_N
```
Again, we choose $u=1+x^2+2y^2$ as the exact solution and adjust $f, g,$ and $u_D$ accordingly
```{math}
f(x,y)&=-6,\\
g(x,y)&=\begin{cases}
0, & y=0,\\
-4, & y=1,
\end{cases}\\
u_D(x,y)&=1+x^2+2y^2.
```
For the ease of programming, we define $g$ as a function over the whole domain $\Omega$ such that $g$ takes on the correct values at $y=0$ and $y=1$. One possible extension is
```{math}
 g(x,y)=-4y.
```
## The variational formulation
The first task is to derive the variational formulatin. This time we cannot omit the boundary term arising from integration by parts, because $v$ is only zero on $\Lambda_D$. We have
```{math}
    -\int_\Omega (\nabla^2u)v\mathrm{d} x =
    \int_\Omega \nabla u \cdot \nabla v \mathrm{d} x
     - \int_{\partial\Omega}\frac{\partial u}{\partial n}v\mathrm{d}s,
```
and since $v=0$ on $\Lambda_D$,
```{math}
  - \int_{\partial\Omega}\frac{\partial u}{\partial n}v\mathrm{d}s=
    - \int_{\Lambda_N}\frac{\partial u}{\partial n}v\mathrm{d}s
    =\int_{\Lambda_N} gv\mathrm{d}s,
```
by applying the boundary condition on $\Lambda_N$.
The resulting weak from reads
```{math}
    \int_\Omega \nabla u \cdot \nabla v \mathrm{d} x = \int_\Omega fv\mathrm{d} x - \int_{\Lambda_N}gv\mathrm{d}s.
```
Expressing this equation in the standard notation $a(u,v)=L(v)$ is straight-forward with 
```{math}
    a(u,v) &= \int_{\Omega}\nabla u \cdot \nabla v \mathrm{d} x,
    L(v) &= \int_{\Omega} fv\mathrm{d} x - \int_{\Lambda_N} gv\mathrm{d}s.
```