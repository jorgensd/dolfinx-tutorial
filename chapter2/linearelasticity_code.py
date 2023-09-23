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
# Author: Jørgen S. Dokken
#
# In this tutorial, you will learn how to:
# - Use a vector function space
# - Create a constant boundary condition on a vector space
# - Visualize cell wise constant functions
# - Compute Von Mises stresses
#
# ## Test problem
# As a test example, we will model a clamped beam deformed under its own weigth in 3D. This can be modeled, by setting the right-hand side body force per unit volume to $f=(0,0,-\rho g)$ with $\rho$ the density of the beam and $g$ the acceleration of gravity. The beam is box-shaped with length $L$ and has a square cross section of width $W$. we set $u=u_D=(0,0,0)$ at the clamped end, x=0. The rest of the boundary is traction free, that is, we set $T=0$. We start by defining the physical variables used in the program.

# Scaled variable
import pyvista
from dolfinx import mesh, fem, plot, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
import numpy as np
L = 1
W = 0.2
mu = 1
rho = 1
delta = W / L
gamma = 0.4 * delta**2
beta = 1.25
lambda_ = beta
g = gamma

# We then create the mesh, which will consist of hexahedral elements, along with the function space. We will use the convenience function `VectorFunctionSpace`. However, we also could have used `ufl`s functionality, creating a vector element `element = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 1)
# `, and intitializing the function space as `V = dolfinx.fem.FunctionSpace(mesh, element)`.

domain = mesh.create_box(MPI.COMM_WORLD, [np.array([0, 0, 0]), np.array([L, W, W])],
                         [20, 6, 6], cell_type=mesh.CellType.hexahedron)
V = fem.VectorFunctionSpace(domain, ("Lagrange", 1))


# ## Boundary conditions
# As we would like to clamp the boundary at $x=0$, we do this by using a marker function, which locate the facets where $x$ is close to zero by machine prescision.

# +
def clamped_boundary(x):
    return np.isclose(x[0], 0)


fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, clamped_boundary)

u_D = np.array([0, 0, 0], dtype=default_scalar_type)
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)
# -

# As we want the traction $T$ over the remaining boundary to be $0$, we create a `dolfinx.Constant`

T = fem.Constant(domain, default_scalar_type((0, 0, 0)))

# We also want to specify the integration measure $\mathrm{d}s$, which should be the integral over the boundary of our domain. We do this by using `ufl`, and its built in integration measures

ds = ufl.Measure("ds", domain=domain)


# ## Variational formulation
# We are now ready to create our variational formulation in close to mathematical syntax, as for the previous problems.

# +
def epsilon(u):
    return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)


def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)


u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(domain, default_scalar_type((0, 0, -rho * g)))
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds
# -

# ```{note}
# Note that we used `nabla_grad` and optionally `nabla_div` for the variational formulation, as oposed to our previous usage of
# `div` and `grad`. This is because for scalar functions $\nabla u$ has a clear meaning
# $\nabla u = \left(\frac{\partial u}{\partial x}, \frac{\partial u}{\partial y}, \frac{\partial u}{\partial z} \right)$.
#
# However, if $u$ is vector valued, the meaning is less clear. Some sources define $\nabla u$ as a matrix with the elements $\frac{\partial u_j}{\partial x_i}$, while other  sources prefer
# $\frac{\partial u_i}{\partial x_j}$. In DOLFINx `grad(u)` is defined as the amtrix with element $\frac{\partial u_i}{\partial x_j}$. However, as it is common in continuum mechanics to use the other definition, `ufl` supplies us with `nabla_grad` for this purpose.
# ```
#
# ## Solve the linear variational problem
# As in the previous demos, we assemble the matrix and right hand side vector and use PETSc to solve our variational problem

problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

# ## Visualization

# As in the previous demos, we can either use Pyvista or Paraview for visualization. We start by using Pyvista. Instead of adding scalar values to the grid, we add vectors.

# +
pyvista.start_xvfb()

# Create plotter and pyvista grid
p = pyvista.Plotter()
topology, cell_types, geometry = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# Attach vector values to grid and warp grid by vector
grid["u"] = uh.x.array.reshape((geometry.shape[0], 3))
actor_0 = p.add_mesh(grid, style="wireframe", color="k")
warped = grid.warp_by_vector("u", factor=1.5)
actor_1 = p.add_mesh(warped, show_edges=True)
p.show_axes()
if not pyvista.OFF_SCREEN:
    p.show()
else:
    figure_as_array = p.screenshot("deflection.png")
# -

# We could also use Paraview for visualizing this.
# As explained in previous sections, we save the solution with `XDMFFile`.
# After opening the file `deformation.xdmf` in Paraview and pressing `Apply`, one can press the `Warp by vector button` ![Warp by vector](warp_by_vector.png) or go through the top menu (`Filters->Alphabetical->Warp by Vector`) and press `Apply`. We can also change the color of the deformed beam by changing the value in the color menu ![color](color.png) from `Solid Color` to `Deformation`.

with io.XDMFFile(domain.comm, "deformation.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    uh.name = "Deformation"
    xdmf.write_function(uh)

# ## Stress computation
# As soon as the displacement is computed, we can compute various stress measures. We will compute the von Mises stress defined as $\sigma_m=\sqrt{\frac{3}{2}s:s}$ where $s$ is the deviatoric stress tensor $s(u)=\sigma(u)-\frac{1}{3}\mathrm{tr}(\sigma(u))I$.

s = sigma(uh) - 1. / 3 * ufl.tr(sigma(uh)) * ufl.Identity(len(uh))
von_Mises = ufl.sqrt(3. / 2 * ufl.inner(s, s))

# The `von_Mises` variable is now an expression that must be projected into an appropriate function space so that we can visualize it. As `uh` is a linear combination of first order piecewise continuous functions, the von Mises stresses will be a cell-wise constant function.

V_von_mises = fem.FunctionSpace(domain, ("DG", 0))
stress_expr = fem.Expression(von_Mises, V_von_mises.element.interpolation_points())
stresses = fem.Function(V_von_mises)
stresses.interpolate(stress_expr)

# In the previous sections, we have only visualized first order Lagrangian functions. However, the Von Mises stresses are piecewise constant on each cell. Therefore, we modify our plotting routine slightly. The first thing we notice is that we  now set values for each cell, which has a one to one correspondence with the degrees of freedom in the function space.

warped.cell_data["VonMises"] = stresses.vector.array
warped.set_active_scalars("VonMises")
p = pyvista.Plotter()
p.add_mesh(warped)
p.show_axes()
if not pyvista.OFF_SCREEN:
    p.show()
else:
    stress_figure = p.screenshot(f"stresses.png")
