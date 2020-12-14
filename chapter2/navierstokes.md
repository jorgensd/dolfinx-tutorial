# The Navier-Stokes equations
Authors: Anders Logg and Hans Petter Langtangen

In this section, we will solve the incompressible Navier-Stokes equations. This problem combines many of the challenges from our previously studied problems: time-dependencies, nonlinearity, and vector-valued variables.

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
The parameter $\mu$ is the dynamic viscosity. Note that the momementum equation [](navier-stokes) is very similar to the elasticity equation [](elasticity-PDE). The difference is in the two additional terms $\rho\left(\frac{\partial u}{\partial t}+ u\cdot \nabla u\right)$ and the different expression for the stress tensor. The two extra terms express the acceleration balanced by the force $F=\nabla \cdot \sigma + f$ per unit volume in Newton's second law of motion.

## Variational formulation
The Navier-Stokes equationsare different from the time-dependent heat equation in that we need to solve a system of equations and ths system of a special type. If we apply the same technique as for the heat equation; that is, replacing the time derivative with a simple difference quotient, we obtain a nonlinear system of equations. This in itself is not a problem as we saw for the [non-linear Poisson equation](./nonlinpoisson.md), but the system has a so-called *saddle point structure* and requires special techniques (special preconditioners and iterative methods) to be solved efficiently.

Instead we will apply a simpler and often very efficient approach, known as a *splitting method*. The idea is to consider the two equations in [](navier-stokes) separately. There exist many splitting strategies for the incompressible Navier-Stokes equations. One of the oldest is the method proposed by Chorin {cite}`chorin1968numerical` and Temam {cite}`Temam1969`, often referred to as *Chorin's method*. We will use a modified version of Chorin's method, the so-called incremental pressure correction scheme (IPCS) due to {cite}`goda1979multistep` which gives improved accuracy compared to the original scheme at little extra cost.

The IPCS scheme involves three steps. First, we compute a *tentative velocity $u$*

## References
```{bibliography} bib_ns.bib
```