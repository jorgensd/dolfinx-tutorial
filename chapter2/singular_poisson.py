# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.18.1
# ---

# # Singular Poisson problem
# Author: Jørgen S. Dokken
#
# In this example, we will solve the singular Poisson problem by attaching information about the
# nullspace of the discretized problem to the matrix system.

# The problem is defined as
#
# \begin{align}
#    -\Delta u &= f &&\text{in } \Omega,\\
#    -\nabla u \cdot \mathbf{n} &= \mathbf{g} &&\text{on } \partial\Omega.
# \end{align}
#
# This problem has a nullspace, i.e. if we take a solution of the problem above, say $\tilde u$ and
# add a constant $c$ to it, $u_c=\tilde u + c$, we still have a solution to the problem.

# We will use a manufactured solution on a unit square to investigate this problem, namely
#
# \begin{align}
#  u(x, y) &= \sin(2\pi x)\\
#  f(x, y) &= -4\pi^2\sin(2\pi x)\\
#  g(x, y) &=
#  \begin{cases}
#    -2\pi  & \text{if } x=0,\\
#    2\pi & \text{if } x=1,\\
#    0 & \text{otherwise.}
#  \end{cases}
# \end{align}

# As we have discretized the Poisson problem in other tutorials, we create a simple wrapper function to set up the variational problem,
# given a manufactured solution

# +
import dolfinx.fem.petsc
from mpi4py import MPI
import numpy as np
import typing
import ufl


def u_ex(mod, x):
    return mod.sin(2 * mod.pi * x[0])


def setup_problem(
    N: int,
) -> typing.Tuple[dolfinx.fem.FunctionSpace, dolfinx.fem.Form, dolfinx.fem.Form]:
    """Set up bilinear and linear form of the singular Poisson problem

    Args:
        N (int): Number of elements in each direction of the mesh.

    Returns:
        The function space, the bilinear form and the linear form of the problem.

    """

    domain = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, N, N, cell_type=dolfinx.mesh.CellType.quadrilateral
    )
    V = dolfinx.fem.functionspace(domain, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    x = ufl.SpatialCoordinate(domain)
    u_exact = u_ex(ufl, x)
    f = -ufl.div(ufl.grad(u_exact))
    n = ufl.FacetNormal(domain)
    g = -ufl.dot(ufl.grad(u_exact), n)

    F = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    F += ufl.inner(g, v) * ufl.ds
    F -= f * v * ufl.dx
    return V, *dolfinx.fem.form(ufl.system(F))


# -

# With the above convenience function set up, we can now address the nullspace.
# We will use PETSc for this, by attaching additional information to the assembled matrices.
# PETSc has a convenience functon for creating constant nullspaces, which we will use here.

# +
from petsc4py import PETSc

nullspace = PETSc.NullSpace().create(constant=True, comm=MPI.COMM_WORLD)
# -

# ## Direct solver
# We start by considering the singular problem using a direct solver (MUMPS).
# Mumps has some additional options to support singular matrices, which we will use.

petsc_options = {
    "ksp_error_if_not_converged": True,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "ksp_monitor": None,
}

# Next, we set up the the KSP solver

ksp = PETSc.KSP().create(MPI.COMM_WORLD)
ksp.setOptionsPrefix("singular_direct")
opts = PETSc.Options()
opts.prefixPush(ksp.getOptionsPrefix())
for key, value in petsc_options.items():
    opts[key] = value
ksp.setFromOptions()
for key, value in petsc_options.items():
    del opts[key]
opts.prefixPop()

# and we assemble the bilinear and linear forms, and create the matrix `A` and right hand side vector `b`.

V, a, L = setup_problem(40)
A = dolfinx.fem.petsc.assemble_matrix(a)
A.assemble()
b = dolfinx.fem.petsc.assemble_vector(L)
b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
ksp.setOperators(A)

# Next,  We first check that this indeed is the nullspace of `A`, then attach the nullspace to the matrix `A`.

assert nullspace.test(A)
A.setNullSpace(nullspace)

# Then, we can solve the linear system of equations

# +
uh = dolfinx.fem.Function(V)
ksp.solve(b, uh.x.petsc_vec)
uh.x.scatter_forward()

ksp.destroy()
# -

# We can now check the $L^2$-error against the analytical solution


def compute_L2_error(uh: dolfinx.fem.Function) -> float:
    mesh = uh.function_space.mesh
    u_exact = u_ex(ufl, ufl.SpatialCoordinate(mesh))
    error_L2 = dolfinx.fem.form(ufl.inner(uh - u_exact, uh - u_exact) * ufl.dx)
    error_local = dolfinx.fem.assemble_scalar(error_L2)
    return np.sqrt(mesh.comm.allreduce(error_local, op=MPI.SUM))


print(f"Direct solver L2 error {compute_L2_error(uh):.5e}")

# We also check that the mean value of the solution is equal to the mean value of the manufactured solution.

u_exact = u_ex(ufl, ufl.SpatialCoordinate(V.mesh))
ex_mean = V.mesh.comm.allreduce(
    dolfinx.fem.assemble_scalar(dolfinx.fem.form(u_exact * ufl.dx)), op=MPI.SUM
)
approx_mean = V.mesh.comm.allreduce(
    dolfinx.fem.assemble_scalar(dolfinx.fem.form(uh * ufl.dx)), op=MPI.SUM
)
print(f"Mean value of manufactured solution: {ex_mean}")
print(f"Mean value of computed solution (direct solver): {approx_mean}")
assert np.isclose(ex_mean, approx_mean), "Mean values do not match!"

# ## Iterative solver
# We can also solve the problem above using an iterative solver,
# for instance GMRES with AMG preconditioning.
# We therefore select a new set of PETSc options, and create a new KSP solver.

ksp_iterative = PETSc.KSP().create(MPI.COMM_WORLD)
ksp_iterative.setOptionsPrefix("singular_iterative")
petsc_options_iterative = {
    "ksp_error_if_not_converged": True,
    "ksp_monitor": None,
    "ksp_type": "gmres",
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
    "pc_hypre_boomeramg_max_iter": 1,
    "pc_hypre_boomeramg_cycle_type": "v",
    "ksp_rtol": 1.0e-13,
}
opts.prefixPush(ksp_iterative.getOptionsPrefix())
for key, value in petsc_options_iterative.items():
    opts[key] = value
ksp_iterative.setFromOptions()
for key, value in petsc_options_iterative.items():
    del opts[key]
opts.prefixPop()

# Instead of setting the nullspace, we attach it as a near nullspace, for the multigrid preconditioner.

A_iterative = dolfinx.fem.petsc.assemble_matrix(a)
A_iterative.assemble()
A_iterative.setNearNullSpace(nullspace)
ksp_iterative.setOperators(A_iterative)

uh_iterative = dolfinx.fem.Function(V)

ksp_iterative.solve(b, uh_iterative.x.petsc_vec)
uh_iterative.x.scatter_forward()

# For the iterative solver, we subtract the mean value of the approximated solution,
# and add the mean value of manufactured solution before computing the error.


approx_mean = V.mesh.comm.allreduce(
    dolfinx.fem.assemble_scalar(dolfinx.fem.form(uh_iterative * ufl.dx)), op=MPI.SUM
)
print("Mean value of computed solution (iterative solver):", approx_mean)
uh_iterative.x.array[:] += ex_mean - approx_mean
approx_mean = V.mesh.comm.allreduce(
    dolfinx.fem.assemble_scalar(dolfinx.fem.form(uh_iterative * ufl.dx)), op=MPI.SUM
)
print(
    "Mean value of computed solution (iterative solver) post normalization:",
    approx_mean,
)
print(f"Iterative solver L2 error {compute_L2_error(uh_iterative):.5e}")

np.testing.assert_allclose(uh.x.array, uh_iterative.x.array, rtol=1e-10, atol=1e-12)
