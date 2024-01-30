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

# # Weak imposition of Dirichlet conditions for the Poisson problem
# Author: Jørgen S. Dokken
#
# In this section, we will go through how to solve the Poisson problem from the [Fundamentals](fundamentals_code.ipynb) tutorial using Nitsche's method {cite}`Nitsche1971`.
# The idea of weak imposition is that we add additional terms to the variational formulation to impose the boundary condition, instead of modifying the matrix system using strong imposition (lifting).
#
# We start by importing the required modules and creating the mesh and function space for our solution

# +
from dolfinx import fem, mesh, plot, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import numpy
from mpi4py import MPI
from ufl import (Circumradius, FacetNormal, SpatialCoordinate, TrialFunction, TestFunction,
                 div, dx, ds, grad, inner)

N = 8
domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N)
V = fem.FunctionSpace(domain, ("Lagrange", 1))
# -

# Next, we create a function containing the exact solution (which will also be used in the Dirichlet boundary condition) and the corresponding source function for the right hand side. Note that we use `ufl.SpatialCoordinate` to define the exact solution, which in turn is interpolated into `uD` and used to create the source function `f`.

uD = fem.Function(V)
x = SpatialCoordinate(domain)
u_ex =  1 + x[0]**2 + 2 * x[1]**2
uD.interpolate(fem.Expression(u_ex, V.element.interpolation_points()))
f = -div(grad(u_ex))

# As opposed to the first tutorial, we now have to have another look at the variational form.
# We start by integrating the problem by parts, to obtain
# \begin{align}
#     \int_{\Omega} \nabla u \cdot \nabla v~\mathrm{d}x - \int_{\partial\Omega}\nabla u \cdot n v~\mathrm{d}s = \int_{\Omega} f v~\mathrm{d}x.
# \end{align}
# As we are not using strong enforcement, we do not set the trace of the test function to $0$ on the outer boundary.
# Instead, we add the following two terms to the variational formulation
# \begin{align}
#     -\int_{\partial\Omega} \nabla  v \cdot n (u-u_D)~\mathrm{d}s + \frac{\alpha}{h} \int_{\partial\Omega} (u-u_D)v~\mathrm{d}s.
# \end{align}
# where the first term enforces symmetry to the bilinear form, while the latter term enforces coercivity.
# $u_D$ is the known Dirichlet condition, and $h$ is the diameter of the circumscribed sphere of the mesh element.
# We create bilinear and linear form, $a$ and $L$
# \begin{align}
#     a(u, v) &= \int_{\Omega} \nabla u \cdot \nabla v~\mathrm{d}x + \int_{\partial\Omega}-(n \cdot\nabla u) v - (n \cdot \nabla v) u + \frac{\alpha}{h} uv~\mathrm{d}s,\\
#     L(v) &= \int_{\Omega} fv~\mathrm{d}x + \int_{\partial\Omega} -(n \cdot \nabla v) u_D + \frac{\alpha}{h} u_Dv~\mathrm{d}s
# \end{align}

u = TrialFunction(V)
v = TestFunction(V)
n = FacetNormal(domain)
h = 2 * Circumradius(domain)
alpha = fem.Constant(domain, default_scalar_type(10))
a = inner(grad(u), grad(v)) * dx - inner(n, grad(u)) * v * ds
a += - inner(n, grad(v)) * u * ds + alpha / h * inner(u, v) * ds
L = inner(f, v) * dx 
L += - inner(n, grad(v)) * uD * ds + alpha / h * inner(uD, v) * ds

# As we now have the variational form, we can solve the linear problem

problem = LinearProblem(a, L)
uh = problem.solve()

# We compute the error of the computation by comparing it to the analytical solution

error_form = fem.form(inner(uh-uD, uh-uD) * dx)
error_local = fem.assemble_scalar(error_form)
errorL2 = numpy.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))
if domain.comm.rank == 0:
    print(fr"$L^2$-error: {errorL2:.2e}")

# We observe that the $L^2$-error is of the same magnitude as in the first tutorial.
# As in the previous tutorial, we also compute the maximal error for all the degrees of freedom.

error_max = domain.comm.allreduce(numpy.max(numpy.abs(uD.x.array-uh.x.array)), op=MPI.MAX)
if domain.comm.rank == 0:
    print(f"Error_max : {error_max:.2e}")

# We observe that as we weakly impose the boundary condition, we no longer fullfill the equation to machine precision at the mesh vertices. We also plot the solution using `pyvista`

# +
import pyvista
pyvista.start_xvfb()

grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))
grid.point_data["u"] = uh.x.array.real
grid.set_active_scalars("u")
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True, show_scalar_bar=True)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("nitsche.png")
# -

# ```{bibliography}
#    :filter: cited and ({"chapter1/nitsche"} >= docnames)
# ```


