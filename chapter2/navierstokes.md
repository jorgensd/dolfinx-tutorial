# The Navier-Stokes equations
Authors: Anders Logg and Hans Petter Langtangen

Minor modifications: Jørgen S. Dokken

In this section, we will solve the incompressible Navier-Stokes equations. This problem combines many of the challenges from our previously studied problems: time-dependencies, non-linearity, and vector-valued variables.

## The PDE problem

The incompressible Navier-Stokes equations form a system of equations for the velocity $u$ and pressure $p$ in an  incompressible fluid
```{math}
:label: navier-stokes
\rho \left( \frac{\partial u }{\partial t} + u \cdot \nabla u \right) &= \nabla \cdot \sigma (u, p) + f,\\
\nabla \cdot u &= 0 
```
The right-hand side of $f$ is a given force per unit volume and just as for the equations of linear elasticity, $\sigma(u,p)$ denotes the stress tensor, which for a Newtonian fluid is given by
```{math}
:label: navier-stokes-stress
\sigma(u, p)=2\mu \epsilon (u) - pI,
```
where $\epsilon(u)$ is the strain-rate tensor
```{math}
    \epsilon(u)=\frac{1}{2}\left(\nabla u + (\nabla u)^T\right).
```
The parameter $\mu$ is the dynamic viscosity. Note that the momentum equation [](navier-stokes) is very similar to the elasticity equation [](elasticity-PDE). The difference is in the two additional terms $\rho\left(\frac{\partial u}{\partial t}+ u\cdot \nabla u\right)$ and the different expression for the stress tensor. The two extra terms express the acceleration balanced by the force $F=\nabla \cdot \sigma + f$ per unit volume in Newton's second law of motion.

## Variational formulation
The Navier-Stokes equations are different from the time-dependent heat equation in that we need to solve a system of equations and the system of a special type. If we apply the same technique as for the heat equation; that is, replacing the time derivative with a simple difference quotient, we obtain a non-linear system of equations. This in itself is not a problem as we saw for the [non-linear Poisson equation](./nonlinpoisson.md), but the system has a so-called *saddle point structure* and requires special techniques (special preconditioners and iterative methods) to be solved efficiently.

Instead we will apply a simpler and often very efficient approach, known as a *splitting method*. The idea is to consider the two equations in [](navier-stokes) separately. There exist many splitting strategies for the incompressible Navier-Stokes equations. One of the oldest is the method proposed by Chorin {cite}`chorin1968numerical` and Temam {cite}`Temam1969`, often referred to as *Chorin's method*. We will use a modified version of Chorin's method, the so-called incremental pressure correction scheme (IPCS) due to {cite}`goda1979multistep` which gives improved accuracy compared to the original scheme at little extra cost.

The IPCS scheme involves three steps. First, we compute a *tentative velocity $u$* by advancing the momentum equation by a midpoint finite difference scheme in time, but using $p^n$ from the previous interval. We will also linearize the nonlinear convective term by using the known velocity $u^n$ from the previous time step: $u^n\cdot \nabla u^n$. Note that there exists several other methods to linearize this term, such as the Adams-Bashforth method, see {cite}`Guermond1999` and {cite}`QuarteroniSaccoSaleri2010`. The variational problem for the first step is: For the $n+1$th step, find $u^*$ such that
```{math}
:label: ipcs-one
    &\left\langle \rho \frac{u^*-u^n}{\Delta t}, v\right\rangle
    + \left\langle \rho u^n\cdot \nabla u^n, v \right\rangle
    +\left\langle \sigma(u^{n+\frac{1}{2}}, p^n), \epsilon(v)\right\rangle\\
    &+ \left\langle p^n n, v \right\rangle_{\partial\Omega}
    -\left\langle \mu \nabla u^{n+\frac{1}{2}}\cdot n, v \right \rangle_{\partial\Omega}=
    \left\langle f^{n+1}, v \right\rangle.
```
This notation, suitable for problems wit many terms in the variational formulations, requires some explaination. 
First, we use the short-hand notation
```{math}
\langle v, w \rangle = \int_{\Omega} vw~\mathrm{d}x, \qquad
\langle v, w \rangle_{\partial\Omega}=\int_{\partial\Omega}vw~\mathrm{d}s.
```
This allows us to express the variational problem in a more compact way. Second, we use the notation $u^{n+\frac{1}{2}}$. This notation refers to the value of $u$ at the midpoint of the interval, usually approximated by an arithmetic mean:
```{math}
   u^{n+\frac{1}{2}}\approx \frac{u^{n}+ u^{n+1}}{2}.
```
Third, we notice that the variational problem [](ipcs-one) arises from the integration by parts of the term 
$langle -\nabla \cdot \sigma, v\rangle$. Just as for the [linear elasticity problem](./linearelasticity.md), we obtain
```{math}
    \langle -\nabla \cdot \sigma, v\rangle =
    \langle \sigma, \epsilon(v) \rangle 
    - \langle T, v\rangle_{\partial \Omega},
```
where $T=\sigma \cdot n$ is the boundary traction. If we solve a problem with a free boundary, we can take $T=0$ on the boundary. However, if we compute the flow through a channel or a pipe and want to model flow that continues into an "imaginary channel" at the outflow, we need to treat this term with some care. 
The assumption we then can make is that the derivative of the velocity in the direction of the channel is zero at the outflow, corresponding to that the flow is "fully developed" or doesn't change significantly downstream at the outflow.
Doing so, the remaining boundary term at the outflow becomes 
$pn - \mu \nabla u \cdot n$, which is the term appearing in the variational problem [](ipcs-one). Note that this argument and the implementation depends exact on the definition of $\nabla u$, as either the  matrix with components $\frac{\partial u_i}{\partial x_j}$ or $\frac{\partial u_j}{\partial x_i}$.
We here choose the  latter, $\frac{\partial u_j}{\partial x_i}$,
which means that we must use the UFL-operator `nabla_grad`. If we use the operator `grad` and the definition $\frac{\partial u_i}{\partial x_j}$, we must instead keep the terms $pn-\mu(\nabla u)^T \cdot n$.

```{admonition} The usage of "nabla_grad" and "grad"
As mentioned in the note in [Linear elasticity implementation](./linearelasticity_code) the usage of `nabla_grad` and `grad` has to be interpreted with care. For the Navier-Stokes equations it is important to consider the term $u\cdot \nabla u$ which should be interpreted as the vector $w$ with elements
$w_i=\sum_{j}\left(u_j\frac{\partial}{\partial x_j}\right)u_i = \sum_j u_j\frac{\partial u_i}{\partial x_j}$. 
This term can be  implemented in  FEniCSx as either 
`grad(u)*u`, since this expression becomes $\sum_j\frac{\partial u_j}{\partial x_j}u_j$, or as `dot(u, nabla_grad(u))` since this 
expression becomes $\sum_i u_i\frac{\partial u_j}{x_i}$. We will use the notation `dot(u, nabla_grad(u))` below since it corresponds more closely to the standard notation $u\cdot \nabla u$.
```

We now move on to the second step in  our splitting scheme for the incompressible Navier-Stokes equations. In the first step, we computed the *tentative velocity* $u^*$ based on the pressure from the previous time step. 
We may now use the computed tentative velocity to compute the new pressure $p^n$:
```{math}
:label: ipcs-two
    \langle \nabla p^{n+1}, \nabla q \rangle = \langle \nabla p^n, \nabla q\rangle - \frac{\rho}{\Delta t}\langle \nabla \cdot u^*, q\rangle.
```
Note here that $q$ is a scalar-valued test function from the pressure space, whereas the test function $v$ in [](ipcs-one) is a vector-valued test function from the velocity space.

One way to think about this step is to subtract the Navier-Stokes momentum equation [](navier-stokes) expressed in terms of the tentative velocity $u^*$ and the pressure $p^n$ from the momentum equation expressed in terms of the velocity $u^{n+1}$ and pressure $p^{n+1}$. This results in the equation
```{math}
\frac{\rho (u^{n+1}-u^*)}{\Delta t}+\nabla p^{n+1}- \nabla p^n = 0.
```
Taking the divergence and requiring that $\nabla \cdot u^{n+1}=0$ by the Navier-Stokes continuity equation, we obtain the equation
```{math}
:label: ipcs-tmp
 - \frac{\rho \nabla\cdot  u^*}{\Delta t}+ \nabla^2p^{n+1}-\nabla^2p^n=0,
```
which is the Poisson problem for the pressure $p^{n+1} resulting in the variational formulation [](ipcs-two).

Finally, we compute the corrected velocity $u^{n+1}$ from the equation [](ipcs-tmp). Multiplying this equation by a test function $v$, we obtain
```{math}
    \rho \langle (u^{n+1} - u^*), v\rangle= -\Delta t\langle \nabla(p^{n+1}-p^n), v\rangle
```

In summary, we may thus solve the incompressible Navier-Stokes equations efficiently by solving a sequence of three linear variational problems in each step.

## References
```{bibliography}
:filter: docname in docnames
```
