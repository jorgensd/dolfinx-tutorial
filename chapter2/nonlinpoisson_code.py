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

# # Implementation
#
# Author: Jørgen S. Dokken
#
# ## Test problem
# To solve a test problem, we need to choose the right hand side $f$ and the coefficient $q(u)$ and the boundary $u_D$. Previously, we have worked with manufactured solutions that can  be reproduced without approximation errors. This is more difficult in non-linear porblems, and the algebra is more tedious. Howeve, we will utilize UFLs differentiation capabilities to obtain a manufactured solution.
#
# For this problem, we will choose $q(u) = 1 + u^2$ and define a two dimensional manufactured solution that is linear in $x$ and $y$:

# +
import ufl
import numpy

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import mesh, fem, io, nls, log
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

def q(u):
    return 1 + u**2


domain = mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
x = ufl.SpatialCoordinate(domain)
u_ufl = 1 + x[0] + 2 * x[1]
f = - ufl.div(q(u_ufl) * ufl.grad(u_ufl))
# -

# Note that since `x` is a 2D vector, the first component (index 0) resemble $x$, while the second component (index 1) resemble $y$. The resulting function `f` can be directly used in variational formulations in DOLFINx.
#
# As we now have defined our source term and exact solution, we can create the appropriate function space and boundary conditions.
# Note that as we have already defined the exact solution, we only have to convert it to a python function that can be evaluated in the interpolation function. We do this by employing the Python `eval` and `lambda`-functions.

V = fem.FunctionSpace(domain, ("Lagrange", 1))
def u_exact(x): return eval(str(u_ufl))


u_D = fem.Function(V)
u_D.interpolate(u_exact)
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: numpy.full(x.shape[1], True, dtype=bool))
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets))

# We are now ready to define the variational formulation. Note that as the problem is non-linear, we have replace the `TrialFunction` with a `Function`, which serves as the unknown of our problem.

uh = fem.Function(V)
v = ufl.TestFunction(V)
F = q(uh) * ufl.dot(ufl.grad(uh), ufl.grad(v)) * ufl.dx - f * v * ufl.dx

# ## Newton's method
# The next step is to define the non-linear problem. As it is non-linear we will use [Newtons method](https://en.wikipedia.org/wiki/Newton%27s_method).
# For details about how to implement a Newton solver, see [Custom Newton solvers](../chapter4/newton-solver.ipynb).
# Newton's method requires methods for evaluating the residual `F` (including application of boundary conditions), as well as a method for computing the Jacobian matrix. DOLFINx provides the function `NonlinearProblem` that implements these methods. In addition to the boundary conditions, you can supply the variational form for the Jacobian (computed if not supplied), and form and jit parameters, see the [JIT parameters section](../chapter4/compiler_parameters.ipynb).

problem = NonlinearProblem(F, uh, bcs=[bc])

# Next, we use the dolfinx Newton solver. We can set the convergence criterions for the solver by changing the absolute tolerance (`atol`), relative tolerance (`rtol`) or the convergence criterion (`residual` or `incremental`).

solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-6
solver.report = True

# We can modify the linear solver in each Newton iteration by accessing the underlying `PETSc` object.

ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "cg"
opts[f"{option_prefix}pc_type"] = "gamg"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

# We are now ready to solve the non-linear problem. We assert that the solver has converged and print the number of iterations.

log.set_log_level(log.LogLevel.INFO)
n, converged = solver.solve(uh)
assert (converged)
print(f"Number of interations: {n:d}")

# We observe that the solver converges after $8$ iterations.
# If we think of the problem in terms of finite differences on a uniform mesh, $\mathcal{P}_1$ elements mimic standard second-order finite differences, which compute the derivative of a linear or quadratic funtion exactly. Here $\nabla u$ is a constant vector, which is multiplied by $1+u^2$, which is a second order polynomial in $x$ and $y$, which the finite difference operator would compute exactly. We can therefore, even with $\mathcal{P}_1$ elements, expect the manufactured solution to be reproduced by the numerical method. However, if we had chosen a nonlinearity, such as $1+u^4$, this would not be the case, and we would need to verify convergence rates.

# +
# Compute L2 error and error at nodes
V_ex = fem.FunctionSpace(domain, ("Lagrange", 2))
u_ex = fem.Function(V_ex)
u_ex.interpolate(u_exact)
error_local = fem.assemble_scalar(fem.form((uh - u_ex)**2 * ufl.dx))
error_L2 = numpy.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))
if domain.comm.rank == 0:
    print(f"L2-error: {error_L2:.2e}")

# Compute values at mesh vertices
error_max = domain.comm.allreduce(numpy.max(numpy.abs(uh.x.array - u_D.x.array)), op=MPI.MAX)
if domain.comm.rank == 0:
    print(f"Error_max: {error_max:.2e}")
