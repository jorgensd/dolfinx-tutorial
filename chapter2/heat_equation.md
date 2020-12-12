# The heat equation
As a first extension of the Poisson problem from the previous chapter, weconsider the time-dependent heat equation, or the time-dependent diffusion equation. This is the natural extension of the Poisson equation describing the stationary distribution of heat in a body to a time-dependent problem. We will see that by discretizing time into small time intervals and applying standard time-stepping methods, we can solve the heat equation by solvinga sequence of variational problems, much like the one we encountered for the Poisson equation.

## The PDE problem
The model problem for the time-dependent PDE reads
\begin{align}
    \frac{\partial u}{\partial t}&=\nabla^2 u + f && \text{in } \Omega \times (0, T],\\
    u &= u_D && \text{n } \partial\Omega \times (0,T],\\
    u &= u_0 && \text{at } t=0.
\end{align}

Here $u$ varies with space and time, e.g. $u=u(x,y,t)$ if the spatial domain $\Omega$ is two-dimensional. The source function $f$ and the boundary values $u_D$ may also vary with space and time. The initial condition $u_0$ is a function of space only.

## The variational formulation
A straightforward approach to solving time-dependent PDEs by the finite element method is to first discretize the time derivative by a finite difference approximation, which yields a sequence of stationary problems, and then turn each stationary problem into a variational formulation. 
We will let the superscript $n$ denote a quantity at time $t_n$, where $n$ is an integer counting time levels. For example, $u^n$ means $u$ at time level $n$. The first step of a finite difference discretization in time consists of sampling the PDE at some time  level, for instance $t_{n+1}$
\begin{align}
    \left(\frac{\partial u }{\partial t}\right)^{n+1}= \nabla^2 u^{n+1}+ f^{n+1}.
\end{align}
The time-derivative can be  approximated by a difference quotient. For simplicity and stability reasons, we choose a simple backward difference:
\begin{align}
    \left(\frac{\partial u }{\partial t}\right)^{n+1}\approx \frac{u^{n+1}-u^n}{\Delta t},
\end{align}
where $\Delta t$ is the time discretization paramter. Inserting the latter expression into our equation at time step $n+1$ yields
\begin{align}
    \frac{u^{n+1}-u^n}{\Delta t}= \nabla^2 u^{n+1}+ f^{n+1}.
\end{align}
This is our time-discrete version of the heat equation. It is called a \textit{backward Euler} or a \textit{implicit Euler} discretization.

We reorder the equation such that the left-hand side contains the terms with only the unknown $u^{n+1}$ and right-hand side contains only computed terms. The resulting equation is a sequence of stationary problems for $u^{n+1}$, assuming $u^{n}$ is known from the previous time step:
\begin{align}
    u^0&=u_0 &&\\
    u^{n+1}-\Delta t \nabla^2 u^{n+1}&= u^{n} + \Delta t f^{n+1}, && n = 0,1,2,\dots
\end{align}
Given $u_0$, we can solve for $u^0, u^1, u^2$ and so on.

We then in turn use the finite element method. This means that we have to turn the equation into its weak formulation. We multiply by the test-function of $v\in \hat{V}$ and integrate second-order derivatives by parts. we now introduce the symbol $u$ for $u^{n+1}$ and we write the resulting weak formulation as

\begin{align}
    a(u,v)&=L_{n+1}(v),
\end{align}
where 
\begin{align}
    a(u,v)&=\int_{\Omega}(uv + \Delta t \nabla u \cdot \nabla v )\mathrm{d} x\\
    L_{n+1}(v)&=\int_{\Omega} (u^n+\Delta t f^{n+1})\mathrm{d} x.
\end{align}


## Projection or interpolation of the initial condition
In addition to the variational problem to be solved in each  time step, we also need to approximate the initial condition. This equation can also be turned into a variational problem
\begin{align}
    a_0(u,v)&=L_0(V),
\end{algin}
with 
\begin{align}
    a_0(u,v)&=\int_{\Omega}uv\mathrm{d} x,\\
    L_0(v)&=\int_{\Omega}u_0v\mathrm{d} x.
\end{align}
When solving this variational problem $u^0$ becomes the $L^2$-projection of the given initial value $u_0$ into the finite element space. 

The alternative is to construct $u^0$ by just interpolating the intitial value $u_0$. We covered how to use interpolation in dolfin-X in the {doc}`membrane chapter <../chapter1/membrane_code>`.

We can use dolfin-X to either project or interpolate the initial condition. The most common choice is to use an projection, which computes an approximation to $u_0$. However, in some applications where we want to verify the code by reproducing exact solutions, one must use interpolate. In this chapter, we will use such a problem.
