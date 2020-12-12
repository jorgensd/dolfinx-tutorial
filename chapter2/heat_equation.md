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
    \left(\frac{\partual u }{\partial t}\right)^{n+1}= \nabla^2 u^{n+1}+ f^{n+1}.
\end{align}
The time-derivative can be  approximated by a difference quotient. For simplicity and stability reasons, we choose a simple backward difference:
\begin{align}
    \left(\frac{\partual u }{\partial t}\right)^{n+1}\approx \frac{u^{n+1}-u^n}{\Delta t},
\end{align}
where $\Delta t$ is the time discretization paramter. Inserting the latter expression into our equation at time step $n+1$ yields
\begin{align}
    \frac{u^{n+1}-u^n}{\Delta t}= \nabla^2 u^{n+1}+ f^{n+1}.
\end{align}
This is our time-discrete version of the heat equation. It is called a \textit{backward Euler} or a \textit{implicit Euler} discretization.
