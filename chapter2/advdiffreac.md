# TO BE IMPLEMETED: A system of advection-diffusion-reaction equations
Authors: Hans Petter Langtangen and Anders Logg 

Most of the problems we have encountered so far have a common feature: they all invlove models expressed by a single scalar or vector PDE.
In many situations the model is instead expressed as a system of PDEs, describing different quantities possibly govered by (very) different physics. 
As we saw for the Navier-Stokes equations, one way to solve one equation a system of PDEs in FEniCSx is to use a splitting method where we solve one equation at a time and feed the solution from one  equation into the next. However, one of the strengths with FEniCSx  is the ease by which one can instead define variational problems that couple several PDEs into one compound system. In this system, we will look at how to use FEniCSx to write solvers for such a system of coupled PDEs. The goal is to demonstrate how easy it is to implement fully implicit, also known as monolithic, solvers in FEniCSx.

## The PDE problem
Our model problem is the following system of advection-diffusion-reaction equations:
```{math}
:label: adv-diff-reac
\frac{\partial u_1}{\partial t} + w \cdot \nabla u_1 - \nabla \cdot (\epsilon \nabla u_1) &= f_1 - Ku_1 u_2,\\
\frac{\partial u_2}{\partial t} + w \cdot \nabla u_2 - \nabla \cdot (\epsilon \nabla u_2) &= f_2 - Ku_1 u_2,\\
\frac{\partial u_3}{\partial t} + w \cdot \nabla u_3 - \nabla \cdot (\epsilon \nabla u_3) &= f_3 - Ku_1 u_2 - K u_3,\\
```
This system models the chemical reaction between two species $A$ and $B$ in some domain $\Omega$:
```{math}
   A + B \rightarrow C.
```
We assume that the  reaction is *first-order*, meaning that the reaction rate is proportional to concentrations $[A]$ and $[B]$ of the two species $A$ and $B$:
```{math}
\frac{\mathrm{d}}{\mathrm{d}t}[C]= K[A][B].
```
We also assume that the formed species $C$ spontaneously decaas with a rate proportional to the concentration [C]. In the [PDE system](adv-diff-reac), we use the variatbles $u_1, u_2$ and $u_3$ to denote the concentrations of the three species:
```{math}
 u_1=[A],\qquad u_2=[B], \qquad u_3=[C].
```
We see that the chemical reactions are accounted for in the right-hand sides of the PDE system.

The chemical reactions takes part at each point in the domain $\Omega$. 
In addition, we assume that the species $A, B$ and $C$ diffuse throughout the domain with diffusivity $\epsilon$  (the terms $-\nabla \cdot (\epsilon \nabla u_i)$) and are advected with velocity $w$ (the terms $w\cdot \nabla u_i$).

```{admonition} Implementation note
For this demo to work, we need to have checkpointing implemented
```
