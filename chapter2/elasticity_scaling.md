# Scaling
Authors: Anders Logg and Hans Petter Langtangen

It is often advantageous to scale a problem as it reduces the need for setting physical parameters, and one obtains dimensionless numbers that reflect the competition of parameters and physical effects. We develop the code for the original model with dimensions, and run the scaled problem by tweaking parameters appropriately. Scaling reduces the number of active parameters from $6$ to $2$ for the present application.

In Navier's equation for $u$, arising from insertion of $\sigma(u)$ in [](elasticity-PDE),

```{math}
    -(\lambda + \mu)\nabla (\nabla \cdot u) - \mu \nabla^2 u = f,
```
we insert coordinates made dimensionless by $L$, and $\bar{u}=\frac{u}{U}$, which results in the dimensionless governing equations
```{math}
    - \beta \bar{\nabla}(\bar{\nabla}\cdot \bar{u})-\bar{\nabla}^2\bar{u} = \bar{f}, \qquad \bar{f} = (0,0,\gamma)
```
where $\beta = 1+\frac{\lambda}{\mu}$ is a dimensionless elasticity parameter and where
```{math}
    \gamma=\frac{\rho g L^2}{\mu U}
```
is a dimensionless variable reflecting the ratio of the load $\rho g$ and the shear stress term $\mu \nabla^2u \sim \mu \frac{U}{L^2}$ in the PDE.

One option for the scaling is to chose $U$ such that $\gamma$ is of unit size ($U=\frac{\rho g L^2}{\mu}$). However, in elasticity, this leads to displacements of the size of the geometry. This can be achieved by choosing $U$ equal to the maximum deflection of a clamped beam, for which there actually exists a formula: $U=\frac{3}{2} \rho g L^2\frac{\delta^2}{E}$ where $\delta=\frac{L}{W}$ is a parameter reflecting how slender the beam is, and $E$ is the modulus of elasticity. Thus the dimensionless parameter $\delta$ is very important in the problem (as expected $\delta\gg 1$ is what gives beam theory!). Taking $E$ to be of the same order as $\mu$, which in this case and for many materials, we realize that $\gamma \sim \delta^{-2}$ is an appropriate choice. Experimenting with the code to find a displacement that "looks right" in the plots of the deformed geometry, points to $\gamma=0.4\delta^{-2}$ as our final choice of $\gamma$.

The simulation code implements the problem with dimensions and physical parameters $\lambda, \mu, \rho, g, L$ and $W$. However, we can easily reuse this code for a scaled problem: Just set $\mu=\rho=L=1$, $W$ as $W/L(\delta^{-1})$, $g=\gamma$ and $\lambda=\beta$.