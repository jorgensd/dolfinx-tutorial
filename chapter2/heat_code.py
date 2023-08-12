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

# # A known analytical solution
# Author: Jørgen S. Dokken
#
# Just as for the [Poisson problem](./../chapter1/fundamentals_code), we construct a test problem which makes it easy to determine if the calculations are correct.
#
# Since we know that our first-order time-stepping scheme is exact for linear functions, we create a problem which has linear variation in time. We combine this with a quadratic variation in space. Therefore, we choose the analytical solution to be
# \begin{align}
# u = 1 + x^2+\alpha y^2 + \beta t
# \end{align}
# which yields a function whose computed values at the degrees of freedom will be exact, regardless of the mesh size and $\Delta t$ as long as the mesh is uniformly partitioned.
# By inserting this into our original PDE, we find that the right hand side $f=\beta-2-2\alpha$. The boundary value $u_d(x,y,t)=1+x^2+\alpha y^2 + \beta t$ and the initial value $u_0(x,y)=1+x^2+\alpha y^2$.
#
# We start by defining the temporal discretization parameters, along with the parameters for $\alpha$ and $\beta$.

from petsc4py import PETSc
from mpi4py import MPI
import ufl
from dolfinx import mesh, fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
import numpy
t = 0  # Start time
T = 2  # End time
num_steps = 20  # Number of time steps
dt = (T - t) / num_steps  # Time step size
alpha = 3
beta = 1.2

# As for the previous problem, we define the mesh and appropriate function spaces.

# +

nx, ny = 5, 5
domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.triangle)
V = fem.FunctionSpace(domain, ("Lagrange", 1))


# -

# ## Defining the exact solution
# As in the membrane problem, we create a Python-class to resemble the exact solution

class exact_solution():
    def __init__(self, alpha, beta, t):
        self.alpha = alpha
        self.beta = beta
        self.t = t

    def __call__(self, x):
        return 1 + x[0]**2 + self.alpha * x[1]**2 + self.beta * self.t


u_exact = exact_solution(alpha, beta, t)

# ## Defining the boundary condition
# As in the previous chapters, we define a Dirichlet boundary condition over the whole boundary

u_D = fem.Function(V)
u_D.interpolate(u_exact)
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets))

# ## Defining the variational formualation
# As we have set $t=0$ in `u_exact`, we can reuse this variable to obtain $u_n$ for the first time step.

u_n = fem.Function(V)
u_n.interpolate(u_exact)

# As $f$ is a constant independent of $t$, we can define it as a constant.

f = fem.Constant(domain, beta - 2 - 2 * alpha)

# We can now create our variational formulation, with the bilinear form `a` and  linear form `L`.

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
F = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx - (u_n + dt * f) * v * ufl.dx
a = fem.form(ufl.lhs(F))
L = fem.form(ufl.rhs(F))

# ## Create the matrix and vector for the linear problem
# To ensure that we are solving the variational problem efficiently, we will create several structures which can reuse data, such as matrix sparisty patterns. Especially note as the bilinear form `a` is independent of time, we only need to assemble the matrix once.

A = assemble_matrix(a, bcs=[bc])
A.assemble()
b = create_vector(L)
uh = fem.Function(V)

# ## Define a linear variational solver
# We will use [PETSc](https://www.mcs.anl.gov/petsc/) to solve the resulting linear algebra problem. We use the Python-API `petsc4py` to define the solver. We will use a linear solver.

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

# ## Solving the time-dependent problem
# With these structures in place, we crete our time-stepping loop.
# In this loop, we first update the Dirichlet boundary condition by interpolating the updated
# expression `u_exact` into `V`. The next step is to re-assemble the vector `b`, with the update `u_n`.
# Then, we need to apply the boundary condition to this vector. We do this by using the lifting operation,
# which applies the boundary condition such that symmetry of the matrix is preserved.
# Then we solve the problem using PETSc and update `u_n` with the data from `uh`.

for n in range(num_steps):
    # Update Diriclet boundary condition
    u_exact.t += dt
    u_D.interpolate(u_exact)

    # Update the right hand side reusing the initial vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, L)

    # Apply Dirichlet boundary condition to the vector
    apply_lifting(b, [a], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])

    # Solve linear problem
    solver.solve(b, uh.vector)
    uh.x.scatter_forward()

    # Update solution at previous time step (u_n)
    u_n.x.array[:] = uh.x.array

# ## Verifying the numerical solution
# As in the first chapter, we compute the L2-error and the error at the mesh vertices for the last time step.
# to verify our implementation.

# +
# Compute L2 error and error at nodes
V_ex = fem.FunctionSpace(domain, ("Lagrange", 2))
u_ex = fem.Function(V_ex)
u_ex.interpolate(u_exact)
error_L2 = numpy.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form((uh - u_ex)**2 * ufl.dx)), op=MPI.SUM))
if domain.comm.rank == 0:
    print(f"L2-error: {error_L2:.2e}")

# Compute values at mesh vertices
error_max = domain.comm.allreduce(numpy.max(numpy.abs(uh.x.array - u_D.x.array)), op=MPI.MAX)
if domain.comm.rank == 0:
    print(f"Error_max: {error_max:.2e}")
