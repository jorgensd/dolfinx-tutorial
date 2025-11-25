# Monolithic Incompressible Navier–Stokes with CR–P0 (Steady Picard)

---

We solve the steady incompressible Navier–Stokes equations in a 2D channel using a **monolithic** formulation with **Crouzeix–Raviart (CR)** velocity and **DG-0** pressure. Dirichlet velocities on the left, top, and bottom are imposed **weakly via Nitsche’s method**, suitable for the nonconforming CR element.

---

## Key Parameters
- **Velocity space:** $ V_h = [\mathrm{CR}_1]^2 $ (nonconforming, facet-based)  
- **Pressure space:** $ Q_h = \mathrm{DG}_0 $  
- **Linearization:** Picard (frozen convection); start from Stokes baseline  
- **Stabilization:** small pressure mass $ \varepsilon_p \int_\Omega p\,q\,dx $ removes nullspace  
- **Solver:** PETSc GMRES + Jacobi  
- **Output:** XDMF for ParaView  

---

## Governing Equations

$$
-\nu \Delta \mathbf{u} + (\mathbf{u}\cdot\nabla)\mathbf{u} + \nabla p = \mathbf{f}, 
\quad \text{in } \Omega,
$$

$$
\nabla\cdot\mathbf{u} = 0, \quad \text{in } \Omega.
$$

### Boundary Conditions
- Inlet (left): $ \mathbf{u} = \mathbf{u}_{\text{in}} $
- Walls (top/bottom): $ \mathbf{u} = \mathbf{0} $
- Outlet (right): natural (traction-free)

---

## Weak Formulation

For test functions $ \mathbf{v}, q $, find $ (\mathbf{u},p) $ such that:

$$
a_{\text{core}}(\mathbf{u},p;\mathbf{v},q)
 = \nu (\nabla\mathbf{u}, \nabla\mathbf{v})_\Omega
  - (p, \nabla\cdot\mathbf{v})_\Omega
  + (q, \nabla\cdot\mathbf{u})_\Omega,
$$

$$
a_{\text{conv}}(\mathbf{u};\mathbf{v}\mid\mathbf{u}_k)
 = ((\mathbf{u}_k\cdot\nabla)\mathbf{u},\mathbf{v})_\Omega,
$$

$$
a_{\text{pm}}(p;q)
 = \varepsilon_p (p,q)_\Omega.
$$

Right-hand side:
$$
\ell(\mathbf{v}) = (\mathbf{f},\mathbf{v})_\Omega.
$$

---

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

---

## Example Output (ParaView)

![Velocity field](chapter2/open_cavity_velocity_glyphs.png)

**Figure:** Velocity magnitude field with CR–P0 (Stokes baseline).  
Color shows $ |\mathbf{u}| $ and arrows indicate flow direction and speed.  
Inlet on the left, natural outlet on the right.

---

## Run Instructions (Docker)

```bash
docker run --rm -it -v "$PWD":"$PWD" -w "$PWD" \
  ghcr.io/fenics/dolfinx/dolfinx:stable \
  python3 chapter2/monolithic_navierstokes_crp0.py \
  --uin 0.5 --nu 1e-2 --theta 0.5 --alpha 300
