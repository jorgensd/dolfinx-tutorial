# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Solver configuration
# Author: Jørgen S. Dokken
#
# In this section, we will go through how to specify what linear algebra solver we would like to use to solve our PDEs, as well as how to verify the implemenation by considering convergence rates.
#
# $$
# -\Delta u = f \text{ in } \Omega
# $$
# $$
# u = u_D \text{ on } \partial \Omega.
# $$
# Using the manufactured solution $u_D=\cos(2\pi x)\cos(2\pi y)$, we obtain $f=8\pi^2\cos(2\pi x)\cos(2\pi y)$.
# We start by creating a generic module for evaluating the analytical solution  at any point $x$.

# +
from dolfinx.fem import dirichletbc, FunctionSpace, Function, locate_dofs_topological
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_unit_square, locate_entities_boundary

from mpi4py import MPI
from petsc4py import PETSc
from ufl import SpatialCoordinate, TestFunction, TrialFunction, div, dx, inner, grad

import numpy
import ufl


def u_ex(mod):
    return lambda x: mod.cos(2 * mod.pi * x[0]) * mod.cos(2 * mod.pi * x[1])


# -

# Note that the return type of `u_ex` is a `lambda` function. Thus, we can create two different lambda functions, one using `numpy` (which will be used for interpolation) and one using `ufl` (which will be used for defining the source term)

u_numpy = u_ex(numpy)
u_ufl = u_ex(ufl)

# We start by using ufl to define our source term, using `ufl.SpatialCoordinate` as input to `u_ufl`.

mesh = create_unit_square(MPI.COMM_WORLD, 30, 30)
x = SpatialCoordinate(mesh)
f = -div(grad(u_ufl(x)))

# Next, we define our linear variational problem

V = FunctionSpace(mesh, ("Lagrange", 1))
u = TrialFunction(V)
v = TestFunction(V)
a = inner(grad(u), grad(v)) * dx
L = f * v * dx
u_bc = Function(V)
u_bc.interpolate(u_numpy)
facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, lambda x: numpy.full(x.shape[1], True))
dofs = locate_dofs_topological(V, mesh.topology.dim - 1, facets)
bcs = [dirichletbc(u_bc, dofs)]

# We start by solving the problem with an LU factorization, a direct solver method (similar to Gaussian elimination).

default_problem = LinearProblem(a, L, bcs=bcs,
                                petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = default_problem.solve()

# We now look at the solver process by inspecting the `PETSc`-solver. As the view-options in PETSc are not adjusted for notebooks (`solver.view()` will print output to the terminal if used in a `.py` file), we write the solver output to file and read it in and print the output.

lu_solver = default_problem.solver
viewer = PETSc.Viewer().createASCII("lu_output.txt")
lu_solver.view(viewer)
solver_output = open("lu_output.txt", "r")
for line in solver_output.readlines():
    print(line)

# This is a very robust and simple method, and is the recommended method up to a few thousand unknowns and can be efficiently used for many 2D and smaller 3D problems. However, sparse LU decomposition quickly becomes slow, as for a $N\times N$-matrix the number of floating point operations scales as $\sim (2/3)N^3$.
#
# For large problems, we instead need to use an iterative method which are faster and require less memory.
# ## Choosing a linear solver and preconditioner
# As the Poisson equation results in a symmetric, positive definite system matrix, the optimal Krylov solver is the conjugate gradient (Lagrange) method. The default preconditioner is the incomplete LU factorization (ILU), which is a popular and robous overall preconditioner. We can change the preconditioner by setting `"pc_type"` to some of the other preconditioners in petsc, which you can find in at [PETSc KSP solvers](https://petsc.org/release/manual/ksp/#tab-kspdefaults) and [PETSc preconditioners](https://petsc.org/release/manual/ksp/#tab-pcdefaults).
# You can set any opition in `PETSc` through the `petsc_options` input, such as the absolute tolerance (`"ksp_atol"`), relative tolerance (`"ksp_rtol"`) and maximum number of iterations (`"ksp_max_it"`).

cg_problem = LinearProblem(a, L, bcs=bcs,
                           petsc_options={"ksp_type": "cg", "ksp_rtol": 1e-6, "ksp_atol": 1e-10, "ksp_max_it": 1000})
uh = cg_problem.solve()
cg_solver = cg_problem.solver
viewer = PETSc.Viewer().createASCII("cg_output.txt")
cg_solver.view(viewer)
solver_output = open("cg_output.txt", "r")
for line in solver_output.readlines():
    print(line)

# For non-symmetrix problems, a Krylov solver for non-symmetrix systems, such as GMRES is a better.

gmres_problem = LinearProblem(a, L, bcs=bcs,
                              petsc_options={"ksp_type": "gmres", "ksp_rtol": 1e-6, "ksp_atol": 1e-10, "ksp_max_it": 1000, "pc_type": "none"})
uh = gmres_problem.solve()
gmres_solver = gmres_problem.solver
viewer = PETSc.Viewer().createASCII("gmres_output.txt")
gmres_solver.view(viewer)
solver_output = open("gmres_output.txt", "r")
for line in solver_output.readlines():
    print(line)

# ```{admonition} A remark regarding verification using iterative solvers
# When we consider manufactured solutions where we expect the resulting error to be of machine precision, it gets complicated when we use iterative methods. The problem is to keep the error due to the iterative solution smaller than the tolerance used in the iterative test. For linear elements and small meshes, a tolerance of between $10^{-11}$ and $10^{-12}$ works well in the case of Krylov solvers too.
# ```
