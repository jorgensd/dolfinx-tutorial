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

# # Hyperelasticity
# Author: Jørgen S. Dokken and Garth N. Wells
#
# This section shows how to solve the hyperelasticity problem for deformation of a beam.
#
# We will also show how to create a constant boundary condition for a vector function space.
#
# We start by importing DOLFINx and some additional dependencies.
# Then, we create a slender cantilever consisting of hexahedral elements and create the function space `V` for our unknown.

# +
import numpy as np
import ufl

from petsc4py import PETSc
from mpi4py import MPI
from dolfinx import fem, mesh, plot
L = 20.0
domain = mesh.create_box(MPI.COMM_WORLD,[[0.0,0.0,0.0], [L, 1, 1]], [20, 5, 5], mesh.CellType.hexahedron)
V = fem.VectorFunctionSpace(domain, ("CG", 2))


# -

# We create two python functions for determining the facets to apply boundary conditions to

# +
def left(x):
    return np.isclose(x[0], 0)

def right(x):
    return np.isclose(x[0], L)

fdim = domain.topology.dim -1
left_facets = mesh.locate_entities_boundary(domain, fdim, left)
right_facets = mesh.locate_entities_boundary(domain, fdim, right)
# -

# Next, we create a  marker based on these two functions

# Concatenate and sort the arrays based on facet indices. Left facets marked with 1, right facets with two
marked_facets = np.hstack([left_facets, right_facets])
marked_values = np.hstack([np.full_like(left_facets, 1), np.full_like(right_facets, 2)])
sorted_facets = np.argsort(marked_facets)
facet_tag = mesh.meshtags(domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets])

# We then create a function for supplying the boundary condition on the left side, which is fixed.

u_bc = np.array((0,) * domain.geometry.dim, dtype=PETSc.ScalarType)

# To apply the boundary condition, we identity the dofs located on the facets marked by the `MeshTag`.

left_dofs = fem.locate_dofs_topological(V, facet_tag.dim, facet_tag.find(1))
bcs = [fem.dirichletbc(u_bc, left_dofs, V)]

# Next, we define the body force on the reference configuration (`B`), and nominal (first Piola-Kirchhoff) traction (`T`). 

B = fem.Constant(domain, PETSc.ScalarType((0, 0, 0)))
T = fem.Constant(domain, PETSc.ScalarType((0, 0, 0)))

# Define the test and solution functions on the space $V$

v = ufl.TestFunction(V)
u = fem.Function(V)

# Define kinematic quantities used in the problem

# +
# Spatial dimension
d = len(u)

# Identity tensor
I = ufl.variable(ufl.Identity(d))

# Deformation gradient
F = ufl.variable(I + ufl.grad(u))

# Right Cauchy-Green tensor
C = ufl.variable(F.T * F)

# Invariants of deformation tensors
Ic = ufl.variable(ufl.tr(C))
J  = ufl.variable(ufl.det(F))
# -

# Define the elasticity model via a stored strain energy density function $\psi$, and create the expression for the first Piola-Kirchhoff stress:

# Elasticity parameters
E = PETSc.ScalarType(1.0e4)
nu = PETSc.ScalarType(0.3)
mu = fem.Constant(domain, E/(2*(1 + nu)))
lmbda = fem.Constant(domain, E*nu/((1 + nu)*(1 - 2*nu)))
# Stored strain energy density (compressible neo-Hookean model)
psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J))**2
# Stress
# Hyper-elasticity
P = ufl.diff(psi, F)

# ```{admonition} Comparison to linear elasticity
# To illustrate the difference between linear and hyperelasticity, the following lines can be uncommented to solve the linear elasticity problem.
# ```

# +
# P = 2.0 * mu * ufl.sym(ufl.grad(u)) + lmbda * ufl.tr(ufl.sym(ufl.grad(u))) * I
# -

# Define the variational form with traction integral over all facets with value 2. We set the quadrature degree for the integrals to 4.

metadata = {"quadrature_degree": 4}
ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tag, metadata=metadata)
dx = ufl.Measure("dx", domain=domain, metadata=metadata)
# Define form F (we want to find u such that F(u) = 0)
F = ufl.inner(ufl.grad(v), P)*dx - ufl.inner(v, B)*dx - ufl.inner(v, T)*ds(2) 

# As the varitional form is non-linear and written on residual form, we use the non-linear problem class from DOLFINx to set up required structures to use a Newton solver.

problem = fem.petsc.NonlinearProblem(F, u, bcs)

# and then create and customize the Newton solver

# +
from dolfinx import nls
solver = nls.petsc.NewtonSolver(domain.comm, problem)

# Set Newton solver options
solver.atol = 1e-8
solver.rtol = 1e-8
solver.convergence_criterion = "incremental"
# -

# We create a function to plot the solution at each time step.

# +
import pyvista
pyvista.set_jupyter_backend("ipygany")

grid = pyvista.UnstructuredGrid(*plot.create_vtk_mesh(domain, domain.topology.dim))

def plot_function(t, uh):
    """
    Create a figure of the concentration uh warped visualized in 3D at timet step t.
    """
    p = pyvista.Plotter()
    # Create grid defined by the function space for visualization of the function
    topology, cells, geometry = plot.create_vtk_mesh(uh.function_space)
    function_grid = pyvista.UnstructuredGrid(topology, cells, geometry)
    var_name = f"u({t})"
    values = np.zeros((geometry.shape[0], 3))
    values[:, :len(uh)] = uh.x.array.reshape(geometry.shape[0], len(uh))
    function_grid[var_name] = values
    function_grid.set_active_vectors(var_name)
    # Warp mesh by deformation
    warped = function_grid.warp_by_vector(var_name, factor=1)
    
    # Add mesh to plotter and visualize
    actor = p.add_mesh(warped)
    p.show_axes()
    if not pyvista.OFF_SCREEN:
       p.show()
    else:
        pyvista.start_xvfb()
        figure_as_array = p.screenshot(f"diffusion_{t:.2f}.png")
        # Clear plotter for next plot
        p.remove_actor(actor)

plot_function(0, u)
# -

# Finally, we solve the problem over several time steps, updating the y-component of the traction

from dolfinx import log
log.set_log_level(log.LogLevel.INFO)
tval0 = -1.5
for n in range(1, 10):
    T.value[2] = n * tval0
    num_its, converged = solver.solve(u)
    assert(converged)
    u.x.scatter_forward()
    print(f"Time step {n}, Number of iterations {num_its}, Load {T.value}")
    plot_function(n, u)


