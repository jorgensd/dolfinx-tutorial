# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Error control: Computing convergence rates
# Author: Jørgen S. Dokken, Hans Petter Langtangen, Anders Logg
#
# For any numerical method one of the most central questions is its *convergence rate*: How fast does the error go to zero when the resolution is increased (mesh size decreased).
#
# For the finite element method, this usually corresponds to proving, theoretically or imperically, that the error $e=u_e-u_h$ is bounded by the mesh size $h$ to some power $r$, that is $\vert\vert e \vert\vert\leq Ch^r$ for some mesh independent constant $C$. The number $r$ is called the *convergence rate* of the method. Note that the different norms like the $L^2$-norm $\vert\vert e\vert\vert$ or the $H_0^1$-norm have different convergence rates.

# ## Computing error norms
# We start by creating a manufactured problem, using the same problem as in [the solver configuration](./solvers.ipynb).
#

# +
from dolfinx import default_scalar_type
from dolfinx.fem import (Expression, Function, FunctionSpace,
                         assemble_scalar, dirichletbc, form, locate_dofs_topological)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_unit_square, locate_entities_boundary

from mpi4py import MPI
from ufl import SpatialCoordinate, TestFunction, TrialFunction, div, dot, dx, grad, inner

import ufl
import numpy as np


def u_ex(mod):
    return lambda x: mod.cos(2 * mod.pi * x[0]) * mod.cos(2 * mod.pi * x[1])


u_numpy = u_ex(np)
u_ufl = u_ex(ufl)


def solve_poisson(N=10, degree=1):

    mesh = create_unit_square(MPI.COMM_WORLD, N, N)
    x = SpatialCoordinate(mesh)
    f = -div(grad(u_ufl(x)))
    V = FunctionSpace(mesh, ("Lagrange", degree))
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v)) * dx
    L = f * v * dx
    u_bc = Function(V)
    u_bc.interpolate(u_numpy)
    facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, lambda x: np.full(x.shape[1], True))
    dofs = locate_dofs_topological(V, mesh.topology.dim - 1, facets)
    bcs = [dirichletbc(u_bc, dofs)]
    default_problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    return default_problem.solve(), u_ufl(x)


# -

# Now, we can compute the error between the analyical solution `u_ex=u_ufl(x)` and approximated solution `uh`. A natural choice might seem to compute `(u_ex-uh)**2*ufl.dx`.

uh, u_ex = solve_poisson(10)
comm = uh.function_space.mesh.comm
error = form((uh - u_ex)**2 * ufl.dx)
E = np.sqrt(comm.allreduce(assemble_scalar(error), MPI.SUM))
if comm.rank == 0:
    print(f"L2-error: {E:.2e}")

# Sometimes it is of interest to compute the error fo the gradient field, $\vert\vert \nabla(u_e-u_h)\vert\vert$, often referred to as the $H_0^1$-nrom of the error, this can be expressed as

eh = uh - u_ex
error_H10 = form(dot(grad(eh), grad(eh)) * dx)
E_H10 = np.sqrt(comm.allreduce(assemble_scalar(error_H10), op=MPI.SUM))
if comm.rank == 0:
    print(f"H01-error: {E_H10:.2e}")


# ### Reliable error norm computation
# However, as this gets expanded to `u_ex**2 + uh**2 - 2*u_ex*uh`. If the error is small, (and the solution itself is of moderate size), this calculation will correspond to subtract two positive numbers `u_ex**2 + uh**2`$\sim 1$ and `2*u_ex*u`$\sim 1$ yielding a small number, prone to round-off errors.
#
# To avoid this issue, we interpolate the approximate and exact solution into a higher order function space. Then we subtract the degrees of freedom from the interpolated functions to create a new error function. Then, finally, we assemble/integrate the square difference and take the square root to get the L2 norm.

def error_L2(uh, u_ex, degree_raise=3):
    # Create higher order function space
    degree = uh.function_space.ufl_element().degree()
    family = uh.function_space.ufl_element().family()
    mesh = uh.function_space.mesh
    W = FunctionSpace(mesh, (family, degree + degree_raise))
    # Interpolate approximate solution
    u_W = Function(W)
    u_W.interpolate(uh)

    # Interpolate exact solution, special handling if exact solution
    # is a ufl expression or a python lambda function
    u_ex_W = Function(W)
    if isinstance(u_ex, ufl.core.expr.Expr):
        u_expr = Expression(u_ex, W.element.interpolation_points)
        u_ex_W.interpolate(u_expr)
    else:
        u_ex_W.interpolate(u_ex)

    # Compute the error in the higher order function space
    e_W = Function(W)
    e_W.x.array[:] = u_W.x.array - u_ex_W.x.array

    # Integrate the error
    error = form(ufl.inner(e_W, e_W) * ufl.dx)
    error_local = assemble_scalar(error)
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)


# ## Computing convergence rates
# Let us consider a sequence of mesh resolutions $h_0>h_1>h_2$, where $h_i=\frac{1}{N_i}$ we compute the errors for a range of $N_i$s

Ns = [4, 8, 16, 32, 64]
Es = np.zeros(len(Ns), dtype=default_scalar_type)
hs = np.zeros(len(Ns), dtype=np.float64)
for i, N in enumerate(Ns):
    uh, u_ex = solve_poisson(N, degree=1)
    comm = uh.function_space.mesh.comm
    # One can send in either u_numpy or u_ex
    # For L2 error estimations it is reccommended to send in u_numpy
    # as no JIT compilation is required
    Es[i] = error_L2(uh, u_numpy)
    hs[i] = 1. / Ns[i]
    if comm.rank == 0:
        print(f"h: {hs[i]:.2e} Error: {Es[i]:.2e}")

# If we assume that $E_i$ is of the form $E_i=Ch_i^r$, with unknown constants $C$ and $r$, we can compare two consecqutive experiments, $E_{i-1}= Ch_{i-1}^r$ and $E_i=Ch_i^r$, and solve for $r$:
# ```{math}
# r=\frac{\ln(E_i/E_{i-1})}{\ln(h_i/h_{i-1})}
# ```
# The $r$ values should approac the expected convergence rate (which is typically the polynomial degree + 1 for the $L^2$-error.) as $i$ increases. This can be written compactly using `numpy`.

rates = np.log(Es[1:] / Es[:-1]) / np.log(hs[1:] / hs[:-1])
if comm.rank == 0:
    print(f"Rates: {rates}")

# We also do a similar study for different orders of polynomial spaces to verify our previous claim.

degrees = [1, 2, 3, 4]
for degree in degrees:
    Es = np.zeros(len(Ns), dtype=default_scalar_type)
    hs = np.zeros(len(Ns), dtype=np.float64)
    for i, N in enumerate(Ns):
        uh, u_ex = solve_poisson(N, degree=degree)
        comm = uh.function_space.mesh.comm
        Es[i] = error_L2(uh, u_numpy, degree_raise=3)
        hs[i] = 1. / Ns[i]
        if comm.rank == 0:
            print(f"h: {hs[i]:.2e} Error: {Es[i]:.2e}")
    rates = np.log(Es[1:] / Es[:-1]) / np.log(hs[1:] / hs[:-1])
    if comm.rank == 0:
        print(f"Polynomial degree {degree:d}, Rates {rates}")


# ### Infinity norm estimates
# We start by creating a function to compute the infinity norm, the max difference between the approximate and exact solution.

def error_infinity(u_h, u_ex):
    # Interpolate exact solution, special handling if exact solution
    # is a ufl expression or a python lambda function
    comm = u_h.function_space.mesh.comm
    u_ex_V = Function(u_h.function_space)
    if isinstance(u_ex, ufl.core.expr.Expr):
        u_expr = Expression(u_ex, u_h.function_space.element.interpolation_points)
        u_ex_V.interpolate(u_expr)
    else:
        u_ex_V.interpolate(u_ex)
    # Compute infinity norm, furst local to process, then gather the max
    # value over all processes
    error_max_local = np.max(np.abs(u_h.x.array - u_ex_V.x.array))
    error_max = comm.allreduce(error_max_local, op=MPI.MAX)
    return error_max


# Running this for various polynomial degrees yields:

for degree in degrees:
    Es = np.zeros(len(Ns), dtype=default_scalar_type)
    hs = np.zeros(len(Ns), dtype=np.float64)
    for i, N in enumerate(Ns):
        uh, u_ex = solve_poisson(N, degree=degree)
        comm = uh.function_space.mesh.comm
        Es[i] = error_infinity(uh, u_numpy)
        hs[i] = 1. / Ns[i]
        if comm.rank == 0:
            print(f"h: {hs[i]:.2e} Error: {Es[i]:.2e}")
    rates = np.log(Es[1:] / Es[:-1]) / np.log(hs[1:] / hs[:-1])
    if comm.rank == 0:
        print(f"Polynomial degree {degree:d}, Rates {rates}")

# We observe super convergence for second order polynomials, yielding a fourth order convergence.
