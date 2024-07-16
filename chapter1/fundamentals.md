# Solving the Poisson equation

Authors: Hans Petter Langtangen, Anders Logg

Adapted to FEniCSx by Jørgen S. Dokken

The goal of this tutorial is to solve one of the most basic PDEs, the Poisson equation, with a few lines of code in FEniCSx.
We start by introducing some fundamental FEniCSx objects, such as `Function`, `functionspace`, `TrialFunction` and `TestFunction`, and learn how to write a basic PDE solver.
This will include:

- How to formulate a mathematical variational problem
- How to apply boundary conditions
- How to solve the discrete linear system
- How to visualize the solution

The Poisson equation is the following boundary-value problem
\begin{align}
-\nabla^2 u(\mathbf{x}) &= f(\mathbf{x})&&\mathbf{x} \in \Omega\\
u(\mathbf{x}) &= u_D(\mathbf{x})&& \mathbf{x} \in \partial\Omega
\end{align}

Here, $u=u(\mathbf{x})$ is the unknown function, $f=f(\mathbf{x})$ a prescribed function, $\nabla^2$ the Laplace operator
(often written as $\Delta$), $\Omega$ the spatial domain, and $\partial\Omega$ the boundary of $\Omega$.
The Poisson problem, including both the PDE, $-\nabla^2 u = f$, and the boundary condition, $u=u_D$ on $\partial\Omega$, is an example of a _boundary-value problem_, which must be precisely stated before we can start solving it numerically with FEniCSx.

In the two-dimensional space with coordinates $x$ and $y$, we can expand the Poisson equation as

$-\frac{\partial^2 u}{\partial x^2} - \frac{\partial^2 u}{\partial y^2} = f(x,y)$

The unknown $u$ is now a function of two variables, $u=u(x,y)$, defined over the two-dimensional domain $\Omega$.

The Poisson equation arises in numerous physical contexts, including
heat conduction, electrostatics, diffusion of substances, twisting of
elastic rods, inviscid fluid flow, and water waves. Moreover, the
equation appears in numerical splitting strategies for more complicated
systems of PDEs, in particular the Navier--Stokes equations.

Solving a boundary value problem in FEniCSx consists of the following steps:

1. Identify the computational domain $\Omega$, the PDE, and its corresponding boundary conditions and source terms $f$.
2. Reformulate the PDE as a finite element variational problem.
3. Write a Python program defining the computational domain, the boundary conditions, the variational problem, and the source terms, using FEniCSx.
4. Run the Python program to solve the boundary-value problem. Optionally, you can extend the program to derive quantities such as fluxes and averages,
   and visualize the results.

As we have already covered step 1, we shall now cover steps 2-4.

## Finite element variational formulation

FEniCSx is based on the finite element method, which is a general and
efficient mathematical technique for the numerical solution of
PDEs. The starting point for finite element methods is a PDE
expressed in _variational form_. For readers not familiar with variational problems, we suggest reading a proper treatment on the finite element method, as this tutorial is meant as a brief introduction to the subject. See the original tutorial {cite}`FenicsTutorial` (Chapter 1.6.2).

The basic recipe for turning a PDE into a variational problem is:

- Multiply the PDE by a function $v$
- Integrate the resulting equation over the domain $\Omega$
- Perform integration by parts of those terms with second order derivatives

The function $v$ which multiplies the PDE is called a _test function_. The unknown function $u$ that is to be approximated is referred to as a _trial function_.
The terms trial and test functions are used in FEniCSx too. The test and trial functions belong to certain _function spaces_ that specify the properties of the functions.

In the present case, we multiply the Poisson equation by a test function $v$ and integrate over $\Omega$:

$\int_\Omega (-\nabla^2 u) v~\mathrm{d} x = \int_\Omega f v~\mathrm{d} x.$

Here $\mathrm{d} x$ denotes the differential element for integration over the domain $\Omega$. We will later let $\mathrm{d} s$ denote the differential element for integration over $\partial\Omega$, the boundary of $\Omega$.

A rule of thumb when deriving variational formulations is that one tries to keep the order of derivatives of $u$ and $v$ as small as possible.
Here, we have a second-order differential of $u$, which can be transformed to a first derivative by employing the technique of
[integration by parts](https://en.wikipedia.org/wiki/Integration_by_parts).
The formula reads

$-\int_\Omega (\nabla^2 u)v~\mathrm{d}x
= \int_\Omega\nabla u\cdot\nabla v~\mathrm{d}x- 
\int_{\partial\Omega}\frac{\partial u}{\partial n}v~\mathrm{d}s,$

where $\frac{\partial u}{\partial n}=\nabla u \cdot \vec{n}$ is the derivative of $u$ in the outward normal direction $\vec{n}$ on the boundary.

Another feature of variational formulations is that the test function $v$ is required to vanish on the parts of the boundary where the solution $u$ is known. See for instance {cite}`Langtangen_Mardal_FEM_2019`.

In the present problem, this means that $v$ is $0$ on the whole boundary $\partial\Omega$. Thus, the second term in the integration by parts formula vanishes, and we have that

$\int_\Omega \nabla u \cdot \nabla v~\mathrm{d} x = \int_\Omega f v~\mathrm{d} x.$

If we require that this equation holds for all test functions $v$ in some suitable space $\hat{V}$, the so-called _test space_, we obtain a well-defined mathematical problem that uniquely determines the solution $u$ which lies in some function space $V$. Note that $V$ does not have to be the same space as
$\hat{V}$. We call the space $V$ the _trial space_. We refer to the equation above as the _weak form_/_variational form_ of the original boundary-value problem. We now properly state our variational problem:
Find $u\in V$ such that

$\int_\Omega \nabla u \cdot \nabla v~\mathrm{d} x = \int_\Omega f v~\mathrm{d} x\qquad \forall v \in \hat{V}.$

For the present problem, the trial and test spaces $V$ and $\hat{V}$ are defined as
\begin{align}
V&=\{v\in H^1(\Omega) \vert v=u_D&&\text{on } \partial \Omega \},\\
\hat{V}&=\{v\in H^1(\Omega) \vert v=0 &&\text{on } \partial \Omega \}.
\end{align}
In short, $H^1(\Omega)$ is the Sobolev space containing functions $v$ such that $v^2$ and $\vert \nabla v \vert ^2$ have finite integrals over $\Omega$. The solution of the underlying
PDE must lie in a function space where the derivatives are
also continuous, but the Sobolev space $H^1(\Omega)$ allows functions with discontinuous derivatives.
This weaker continuity requirement in our weak formulation (caused by the integration by parts) is of great importance when it comes to constructing the finite element function space. In particular, it allows the use of piecewise polynomial function spaces. This means that the function spaces are constructed
by stitching together polynomial functions on simple domains
such as intervals, triangles, quadrilaterals, tetrahedra and
hexahedra.

The variational problem is a _continuous problem_: it defines the solution $u$ in the infinite-dimensional function space $V$.
The finite element method for the Poisson equation finds an approximate solution of the variational problem by replacing the infinite-dimensional function spaces $V$ and $\hat{V}$ by _discrete_ (finite dimensional) trial and test spaces $V_h\subset V$ and $\hat{V}_h \subset \hat{V}$. The discrete
variational problem reads: Find $u_h\in V_h$ such that
\begin{align}
\int_\Omega \nabla u_h \cdot \nabla v~\mathrm{d} x &= \int_\Omega fv~\mathrm{d} x && \forall v \in \hat{V}_h.
\end{align}
This variational problem, together with suitable definitions of $V_h$ and $\hat{V}_h$ uniquely define our approximate numerical solution of the Poisson equation.
Note that the boundary condition is encoded as part of the test and trial spaces. This might seem complicated at first glance,
but means that the finite element variational problem and the continuous variational problem look the same.

## Abstract finite element variational formulation

We will introduce the following notation for variational problems:
Find $u\in V$ such that
\begin{align}
a(u,v)&=L(v)&& \forall v \in \hat{V}.
\end{align}
For the Poisson equation, we have:
\begin{align}
a(u,v) &= \int_{\Omega} \nabla u \cdot \nabla v~\mathrm{d} x,\\
L(v) &= \int_{\Omega} fv~\mathrm{d} x.
\end{align}
In the literature $a(u,v)$ is known as the _bilinear form_ and $L(v)$ as a _linear form_.
For every linear problem, we will identify all terms with the unknown $u$ and collect them in $a(u,v)$, and collect all terms with only known functions in $L(v)$.

To solve a linear PDE in FEniCSx, such as the Poisson equation, a user thus needs to perform two steps:

1. Choose the finite element spaces $V$ and $\hat{V}$ by specifying the domain (the mesh) and the type of function space (polynomial degree and type).
2. Express the PDE as a (discrete) variational problem: Find $u\in V$ such that $a(u,v)=L(v)$ for all $v \in \hat{V}$.

## References

```{bibliography}
   :filter: cited and ({"chapter1/fundamentals"} >= docnames)
```
