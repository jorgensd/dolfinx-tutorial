# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Newton's method
# Author: Jørgen S. Dokken
#
# Newtons method, as used in [Non-linear Poisson](./../chapter2/nonlinpoisson_code) is a way of solving a non-linear equation as a sequence of linear equations.
#
# Given a function $F:\mathbb{R}^M\mapsto \mathbb{R}^M$, we have that $u_k, u_{k+1}\in \mathbb{R}^M$ is related as:
# $$x_{k+1} = x_{k} - J_F(x_k)^{-1} F(x_k)$$
# where $J_F$ is the Jacobian matrix of $F$.
#
# We can rewrite this equation as $\delta x_k = x_{k+1} - x_{k}$,
# $$J_F(x_k)\delta x_k = - F(x_k)$$
# and
# $$x_{k+1} = x_k + \delta x_k.$$

# ## Problem specification
# We start by importing all packages needed to solve the problem.

from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
import ufl
import numpy as np
import matplotlib.pyplot as plt


# We will consider the following non-linear problem:
#
# $$ u^2 - 2 u = x^2 + 4x + 3 \text{ in } [0,1] $$
# For this problem, we have two solutions, $u=-x-1$, $u=x+3$.
# We define these roots as python functions, and create an appropriate spacing for plotting these soultions.

# +
def root_0(x):
    return 3 + x[0]

def root_1(x):
    return -1 - x[0]

N = 10
roots = [root_0, root_1]
x_spacing = np.linspace(0, 1, N)
# -

# We will start with an initial guess for this problem, $u_0 = 0$.
# Next, we define the mesh, and the appropriate function space and function `uh` to hold the approximate solution.

mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, N)
V = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))
uh = dolfinx.fem.Function(V)

# ## Definition of residual and Jacobian
# Next, we define the variational form, by multiplying by a test function and integrating over the domain $[0,1]$

v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(mesh)
F = uh**2 * v * ufl.dx - 2 * uh * v * ufl.dx - (x[0]**2 + 4*x[0] + 3)* v * ufl.dx
residual = dolfinx.fem.form(F)

# Next, we can define the jacobian $J_F$, by using `ufl.derivative`.

J = ufl.derivative(F, uh)
jacobian = dolfinx.fem.form(J)

# As we will solve this problem in an iterative fashion, we would like to create the sparse matrix and vector containing the residual only once.
# ## Setup of iteration-independent structures

A = dolfinx.fem.petsc.create_matrix(jacobian)
L = dolfinx.fem.petsc.create_vector(residual)

# Next, we create the linear solver and the vector to hold `dx`.

solver = PETSc.KSP().create(mesh.comm)
solver.setOperators(A)
dx = dolfinx.fem.Function(V)

# We would like to monitor the evolution of `uh` for each iteration. Therefore, we get the dof coordinates, and sort them in increasing order.

i = 0
coords = V.tabulate_dof_coordinates()[:, 0]
sort_order = np.argsort(coords)
max_iterations = 25
solutions = np.zeros((max_iterations, len(coords)))
solutions[0] = uh.x.array[sort_order]

# We are now ready to solve the linear problem. At each iteration, we reassemble the Jacobian and residual, and use the norm of the magnitude of the update (`dx`) as a termination criteria. 
# ## The Newton iterations

i = 0
while i < max_iterations:
    # Assemble Jacobian and residual
    L.zeroEntries()
    A.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix(A, jacobian)
    A.assemble()
    dolfinx.fem.petsc.assemble_vector(L, residual)
    # Scale residual by -1
    L.scale(-1)
    L.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

    # Solve linear problem
    solver.solve(L, dx.vector)
    dx.x.scatter_forward()
    # Update u_{i+1} = u_i + delta x_i
    uh.x.array[:] += dx.x.array
    i+=1

    # Compute norm of update
    correction_norm = dx.vector.norm(0)
    print(f"Iteration {i}: Correction norm {correction_norm}")
    if correction_norm < 1e-10:
        break
    solutions[i] = uh.x.array[sort_order]

# We now compute the magnitude of the residual.

dolfinx.fem.petsc.assemble_vector(L, residual)
print(f"Final residual {L.norm(0)}")

# ## Visualization of Newton iterations
# We next look at the evolution of the solutions and the error of the solution when compared to the two exact roots of the problem.

# +
# Plot solution for each of the iterations
fig = plt.figure(figsize=(15,8))
for j, solution in enumerate(solutions[:i]):
    plt.plot(coords[sort_order], solution[sort_order], label=f"Iteration {j}")

# Plot each of the roots of the problem, and compare the approximate solution with each of them
args = ("--go",)
for j, root in enumerate(roots):
    u_ex = root(x)
    L2_error = dolfinx.fem.form(ufl.inner(uh - u_ex, uh - u_ex) * ufl.dx)
    global_L2 = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(L2_error), op=MPI.SUM)
    print(f"L2-error (root {j}) {np.sqrt(global_L2)}")

    kwargs = {} if j==0 else {"label": "u_exact"}
    plt.plot(x_spacing, root(x_spacing.reshape(1,-1)), *args, **kwargs)
plt.grid()
plt.legend()
plt.show()

# -

# # Newton's method with DirichletBC
# In the previous example, we did not consider handling of Dirichlet boundary conditions. 
# For this example, we will consider the [non-linear Poisson](./../chapter2/nonlinpoisson)-problem.

# +
def q(u):
    return 1 + u**2

domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
x = ufl.SpatialCoordinate(domain)
u_ufl = 1 + x[0] + 2*x[1]
f = - ufl.div(q(u_ufl)*ufl.grad(u_ufl))

V = dolfinx.fem.FunctionSpace(domain, ("CG", 1))
u_exact = lambda x: eval(str(u_ufl))
u_D = dolfinx.fem.Function(V)
u_D.interpolate(u_exact)
fdim = domain.topology.dim - 1
domain.topology.create_connectivity(fdim, fdim+1)
boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
bc = dolfinx.fem.dirichletbc(u_D, dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets))

uh = dolfinx.fem.Function(V)
v = ufl.TestFunction(V)
F = q(uh)*ufl.dot(ufl.grad(uh), ufl.grad(v))*ufl.dx - f*v*ufl.dx
J = ufl.derivative(F, uh)
residual = dolfinx.fem.form(F)
jacobian = dolfinx.fem.form(J)
# -


