# The Helmholtz equation
Author: Antonio Baiano Svizzero 
  
The study of computational acoustics is fundamental in fields such as noise, vibration, and harshness (NVH), noise control, and acoustic design. In this chapter, we focus on the theoretical foundations of the Helmholtz equation - valid for noise problems with harmonic time dependency - and its implementation in FEniCSx to compute the sound pressure for any acoustic system.

## The PDE problem
The acoustic Helmholtz equation in its general form reads

$$
\begin{align}
\nabla^2 p + k^2 p = -j \omega \rho_0 q \qquad\text{in } \Omega,
\end{align}
$$

where $k$ is the acoustic wavenumber, $\omega$ is the angular frequency, $j$ the imaginary unit and $q$ is the volume velocity ($m^3/s$) of a generic source field.
In case of a monopole source, we can write  $q=Q \delta(x_s,y_s,z_s)$, where $\delta(x_s,y_s,z_s)$ is the 3D Dirac Delta centered at the monopole location. 

This equation is coupled with the following boundary conditions: 

- Dirichlet BC:  

    $$
    \begin{align}
    p = \bar{p} \qquad \text{on  }  \partial\Omega_p,
    \end{align}
    $$

- Neumann BC:  

    $$
    \begin{align}
    \frac{\partial p}{\partial n} = - j \omega \rho_0 \bar{v}_n\qquad \text{on  }  \partial\Omega_v,
    \end{align}
    $$

- Robin BC:  

    $$
    \begin{align}
    \frac{\partial p}{\partial n} = - \frac{j \omega \rho_0 }{\bar{Z}} p \qquad \text{on  }  \partial\Omega_Z,
    \end{align}
    $$

where we prescribe, respectively, an acoustic pressure $\bar{p}$ on the boundary $\partial\Omega_p$,
a sound particle velocity $\bar{v}_n$ on the boundary $\partial\Omega_v$ and
an acoustic impedance $\bar{Z}$ on the boundary $\partial\Omega_Z$ where $n$ is the outward normal.
In general, any BC can also be frequency dependant, as it happens in real-world applications.

## The variational formulation
Now we have to turn the equation in its weak formulation.
The first step is to multiplicate the equation by a *test function* $v\in \hat V$,
where $\hat V$ is the *test function space*, after which we integrate over the whole domain, $\Omega$:

$$
\begin{align}
\int_{\Omega}\left(\nabla^2 p + k^2 p \right) \bar v ~\mathrm{d}x = -\int_{\Omega} j \omega \rho_0 q \bar v ~\mathrm{d}x.
\end{align}
$$

Here, the unknown function $p$ is referred to as *trial function* and the $\bar{\cdot}$ is the complex conjugate operator.

In order to keep the order of derivatives as low as possible, we use integration by parts on the Laplacian term: 

$$
\begin{align}
\int_{\Omega}(\nabla^2 p) \bar v ~\mathrm{d}x =
-\int_{\Omega} \nabla p  \cdot \nabla \bar v ~\mathrm{d}x
+ \int_{\partial \Omega} \frac{\partial p}{\partial n} \bar v ~\mathrm{d}s.
\end{align}
$$

Substituting in the original version and rearranging we get: 

$$
\begin{align}
\int_{\Omega} \nabla p  \cdot \nabla \bar v ~\mathrm{d}x
- k^2 \int_{\Omega} p \bar v ~\mathrm{d} x = \int_{\Omega} j \omega \rho_0 q \bar v ~\mathrm{d}x
+ \int_{\partial \Omega} \frac{\partial p}{\partial n} \bar v ~\mathrm{d}s.
\end{align}
$$

Since we are dealing with complex values, the inner product in the first equation is *sesquilinear*,
meaning it is linear in one argument and conjugate-linear in the other,
as explained in [The Poisson problem with complex numbers](../chapter1/complex_mode).

The last term can be written using the Neumann and Robin BCs, that is: 

$$
\begin{align}
\int_{\partial \Omega} \frac{\partial p}{\partial n} \bar v ~\mathrm{d}s =
-\int_{\partial \Omega_v}  j \omega \rho_0  \bar{v}_n \bar v ~\mathrm{d}s
- \int_{\partial \Omega_Z}  \frac{j \omega \rho_0}{\bar{Z}} p \bar v ~\mathrm{d}s.
\end{align}
$$

Substituting, rearranging and taking out of integrals the terms with $j$ and $\omega$ we get the variational formulation of the Helmholtz.
Find $u \in V$ such that: 

$$
\begin{align}
\int_{\Omega} \nabla p  \cdot \nabla \bar v ~\mathrm{d}x
+ \frac{j \omega }{\bar{Z}} \int_{\partial \Omega_Z}   \rho_0 p \bar v ~\mathrm{d}s 
- k^2 \int_{\Omega} p \bar v ~\mathrm{d}x
= j \omega \int_{\Omega}  \rho_0 q \bar v ~\mathrm{d}x
-j \omega\int_{\partial \Omega_v}   \rho_0 \bar{v}_n \bar v ~\mathrm{d}s \qquad  \forall v \in \hat{V}.
\end{align}
$$

We define the sesquilinear form $a(p,v)$ is

$$
\begin{align}
a(p,v) = \int_{\Omega} \nabla p  \cdot \nabla \bar v ~\mathrm{d}x
+ \frac{j \omega }{\bar{Z}} \int_{\partial \Omega_Z}  \rho_0  p \bar v ~\mathrm{d}s
- k^2 \int_{\Omega} p \bar v ~\mathrm{d}x 
\end{align}
$$

and the linear form $L(v)$ reads

$$
\begin{align}
L(v) =  j \omega \int_{\Omega}\rho_0 q \bar v ~\mathrm{d}x - j \omega \int_{\partial \Omega_v}  \rho_0 \bar{v}_n \bar v ~\mathrm{d}s.
\end{align}
$$
