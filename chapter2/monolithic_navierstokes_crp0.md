# Monolithic Incompressible Navier–Stokes with CR–P0 (Steady Picard)

We solve the steady incompressible Navier–Stokes equations in a 2D channel.

We use a **monolithic** formulation with  
[Crouzeix–Raviart (CR)](https://defelement.org/elements/crouzeix-raviart.html)  
for the velocity and **DG–0** for the pressure.

Dirichlet conditions for the velocity on the **top**, **bottom**, and **left** boundaries  
are imposed **weakly using Nitsche’s method**  
(see the tutorial on  
[Weak imposition of Dirichlet conditions for the Poisson equation](../chapter1/nitsche)  
for details).

## Key Parameters

- **Velocity space:** $ V_h = [\mathrm{CR}_1]^2 $ (nonconforming, facet-based)  
- **Pressure space:** $ Q_h = \mathrm{DG}_0 $  
- **Linearization:** Picard (frozen convection); start from Stokes baseline  
- **Stabilization:** small pressure mass $ \varepsilon_p \int_\Omega p\,q\,dx $ removes nullspace  
- **Solver:** PETSc GMRES + Jacobi  
- **Output:** {py:class}`VTKWriter<dolfinx.io.VTXWriter>` for ParaView

## Governing Equations

\begin{align}
 -\nu \Delta \mathbf{u} + (\mathbf{u}\cdot\nabla)\mathbf{u} + \nabla p &= \mathbf{f} && \text{in } \Omega,\\
 \nabla\cdot\mathbf{u} &= 0 && \text{in } \Omega.
\end{align}

### Boundary Conditions

- Inlet (left): $ \mathbf{u} = \mathbf{u}_{\text{in}} $  
- Walls (top/bottom): $ \mathbf{u} = \mathbf{0} $  
- Outlet (right): natural (traction-free)

## Weak Formulation

For test functions $ \mathbf{v}, q $, find $ (\mathbf{u},p) $ such that:

\begin{align}
a_{\text{core}}((\mathbf{u},p);(\mathbf{v},q))
  &= \nu (\nabla\mathbf{u}, \nabla\mathbf{v})_\Omega
     - (p, \nabla\cdot\mathbf{v})_\Omega
     + (q, \nabla\cdot\mathbf{u})_\Omega,\\
a_{\text{conv}}(\mathbf{u};\mathbf{v}\mid\mathbf{u}_k)
  &= ((\mathbf{u}_k\cdot\nabla)\mathbf{u},\mathbf{v})_\Omega,\\
a_{\text{pm}}(p;q)
  &= \varepsilon_p (p,q)_\Omega.
\end{align}

Right-hand side:
$$
\ell(\mathbf{v}) = (\mathbf{f},\mathbf{v})_\Omega.
$$

> **Reference:**  
> The CR–P0 mixed finite element formulation follows the framework in  
> *F. Brezzi and M. Fortin, Mixed and Hybrid Finite Element Methods, Springer, 1991.*  
> The weak enforcement of velocity Dirichlet conditions is based on the symmetric  
> Nitsche method (*J. Nitsche, 1971*).

### Nitsche Boundary Terms

$$
\begin{aligned}
& -\nu\int_{\Gamma_D} (\nabla\mathbf{u}\,\mathbf{n})\cdot\mathbf{v}\,ds
 -\nu\int_{\Gamma_D} (\nabla\mathbf{v}\,\mathbf{n})\cdot(\mathbf{u}-\mathbf{u}_D)\,ds
 + \alpha\frac{\nu}{h}\int_{\Gamma_D} (\mathbf{u}-\mathbf{u}_D)\cdot\mathbf{v}\,ds \\
&\quad - \int_{\Gamma_D} p\,\mathbf{n}\cdot\mathbf{v}\,ds
 + \int_{\Gamma_D} q\,\mathbf{n}\cdot(\mathbf{u}-\mathbf{u}_D)\,ds.
\end{aligned}
$$

### Under-relaxed Picard Update

$$
\mathbf{u}_{k+1}^{(\text{lin})}
= \theta\,\mathbf{u}_{k}^{(\text{new})}
 + (1-\theta)\,\mathbf{u}_{k}^{(\text{lin})}, \quad 0<\theta\le1.
$$

## Example Output (ParaView)

![Velocity field](open_cavity_velocity_glyphs.png)

**Figure:** Velocity magnitude field with CR–P0 (Stokes baseline).  
Color shows $ |\mathbf{u}| $ and arrows indicate flow direction and speed.  
Inlet on the left, natural outlet on the right.

## Run Instructions (Docker)

```bash
docker run --rm -it -v "$PWD":"$PWD" -w "$PWD" \
  ghcr.io/fenics/dolfinx/dolfinx:stable \
  python3 chapter2/monolithic_navierstokes_crp0.py \
  --uin 0.5 --nu 1e-2 --theta 0.5 --alpha 300
