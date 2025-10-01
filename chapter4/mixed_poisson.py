# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.3
# ---

# # Mixed Poisson with a Schur complement pre-conditioner
# This example demonstrates how to use PETSC fieldsplits with custom preconditions in DOLFINx.
# This example is heavily insipired by the [FEniCSx PCTools example](https://rafinex-external-rifle.gitlab.io/fenicsx-pctools/demo/demo_mixed-poisson.html)
# which was presented in {cite}`rehor2025pctools`.

# We start with the mixed formulation of the Poisson equation, which is given by
# \begin{align}
# \sigma - \nabla u &= 0&&\text{in } \Omega,\\
# \nabla \cdot \sigma &= -f&&\text{in } \Omega,\\
# u &= u_D &&\text{on } \Gamma_D,\\
# \sigma \cdot n &= g &&\text{on } \Gamma_N,
# \end{align}
#
# As in previous examples, we pick a manufactured solution to ensure that we can verify
# the correctness of our implementation.
# The manufactured solution is given by
# \begin{align}
# u_{ex}(x, y) &= \sin(\pi x) + y^2.
# \end{align}


def u_ex(mod, x):
    return mod.sin(mod.pi * x[0]) + x[1] ** 2


# We choose to solve the problem on a unit square,

# +
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx

N = 400
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
# -

# where $\Gamma_D = \{(x, 0) \vert x \in [0, 1]\}\cup\{ (x, 1) \vert x \in [0, 1]\}$
# and $\Gamma_N = \{(0, y) \vert y \in [0, 1]\}\cup\{(1, y) \vert y \in [0, 1]\}$.

# +

import numpy as np


def Gamma_D(x):
    return (
        np.isclose(x[1], 0)
        | np.isclose(x[1], 1)
        | np.isclose(x[0], 0)
        | np.isclose(x[0], 1)
    )


def Gamma_N(x):
    return np.full_like(
        x[0], 0, dtype=bool
    )  # np.isclose(x[0], 0) | np.isclose(x[0], 1)


# -

# We define the function space for the vector-valued flux $p\in Q$ as the zeroth order discontinuous Lagrange space,
# while the scalar potential $u \in V$ is defined in first order
# [Brezzi-Douglas-Marini space](https://defelement.org/elements/brezzi-douglas-marini.html).

V = dolfinx.fem.functionspace(mesh, ("DG", 0))
Q = dolfinx.fem.functionspace(mesh, ("BDM", 1))

# We define a `ufl.MixedFunctionSpace` to automatically handle the block structure of the problem

# +
import ufl

W = ufl.MixedFunctionSpace(*[Q, V])
# -

# Next, we have to define the bilinear and linear forms.
# We do this as usual, by introducing a test functions $v\in V$ and $\tau\in Q$ and a trial function $u\in V$ and $q\in Q$,
# and integrate the first equation by parts.
#
# \begin{align}
# \int_\Omega \sigma \cdot \tau - \nabla u \cdot \tau ~\mathrm{d} x &=
# \int_\Omega \sigma \cdot \tau + u \nabla \cdot \tau ~\mathrm{d} x
# - \sum_{f_i\in \mathit{Fi}}\int_{f_i}\left[u\right] \tau \cdot \mathbf{n}_i~\mathrm{d}s
# - \int_{\partial\Omega} u \tau\cdot \mathbf{n}~\mathrm{d}s,\\
# &=\int_\Omega \sigma \cdot \tau + u \nabla \cdot \tau ~\mathrm{d} x
# - \int_{\Gamma_D} u_D \tau\cdot \mathbf{n}~\mathrm{d}s,\\
# \end{align}
#
# where $f_i$ is an interior facet of the mesh, $\mathbf{n}_i$ is an outwards pointing normal of one of the two
# adjacent elements. We will enforce the boundary conditions strongly by using a
# `dolfinx.fem.dirichletbc` on both $\Gamma_N$, which makes its integral dissapear, while we enforced the Dirichlet boundary condition
# on $\Gamma_D$ weakly._
# We enforce the continuity of $u$ weakly by removing the jump term.
# Thus we end up with:
#
# Find $u\in V_{u_D}, \sigma \in Q_{g}$ such that
#
# \begin{align}
# \begin{split}
# \int_\Omega \sigma \cdot \tau + u \nabla \cdot \tau ~\mathrm{d} x&= \int_{\Gamma_D} u_D \tau\cdot \mathbf{n}~\mathrm{d}s,\\\\
# \int_\Omega \nabla \cdot \sigma v ~\mathrm{d} x&=-\int_\Omega f v ~\mathrm{d} x
# \end{split}\qquad \forall v \in V_{0}, \tau \in Q_{0}
# \end{align}

u_D = dolfinx.fem.Function(V)
u_D.interpolate(lambda x: u_ex(np, x))
mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)
gamma_d_facets = dolfinx.mesh.locate_entities_boundary(
    mesh, mesh.topology.dim - 1, Gamma_D
)
tag = 3
ft = dolfinx.mesh.meshtags(
    mesh,
    mesh.topology.dim - 1,
    gamma_d_facets,
    np.full(gamma_d_facets.shape[0], tag, dtype=np.int32),
)
dGammaD = ufl.Measure("ds", domain=mesh, subdomain_data=ft, subdomain_id=tag)


sigma, u = ufl.TrialFunctions(W)
tau, v = ufl.TestFunctions(W)
n = ufl.FacetNormal(mesh)
a = ufl.inner(sigma, tau) * ufl.dx
a += u * ufl.div(tau) * ufl.dx
a += ufl.inner(ufl.div(sigma), v) * ufl.dx

# This can be split into a saddle point problem, with discretized matrices $A$ and $B$ and discretized
# right-hand side $\mathbf{b}$.
# \begin{align}
# \begin{pmatrix}
# A & B^T\\
# B & 0
# \end{pmatrix}
# \begin{pmatrix}
# u_h\\
# \sigma_h
# \end{pmatrix}
# = \begin{pmatrix}
# b_0\\
# b_1
# \end{pmatrix}
# \end{align}
# We can extract the block structure of the bilinear form using `ufl.extract_blocks`, which returns a nested list of bilinear forms.
# You can also build this nested list by hand if you want to, but it is usually more error-prone.

# +
a_blocked = ufl.extract_blocks(a)

x = ufl.SpatialCoordinate(mesh)
u_exact = u_ex(ufl, x)
sigma_exact = ufl.grad(u_exact)
f = -ufl.div(sigma_exact)
L = ufl.inner(u_D, ufl.dot(tau, n)) * dGammaD - ufl.inner(f, v) * ufl.dx
L_blocked = ufl.extract_blocks(L)
# -

# Next we create the Dirichlet boundary condition for $\sigma$.
# As we are using manufactured solutions for this problem, we could manually derive the explicit expression
# for $\sigma$ on the boundary $\Gamma_N$.
# However, in general this is not possible (especially for curved boundaries), and we have to use a more generic approach.
# For this we will use the `dolfinx.fem.Expression` class to interpolate the expression into the function space $Q$.
# This is done by evaluating the expression at the physical interpolation points of the mesh.
# A convenience function for this is provided in the `interpolate_facet_expression` function below.

import numpy.typing as npt
import basix.ufl


def interpolate_facet_expression(
    Q: dolfinx.fem.FunctionSpace,
    expr: ufl.core.expr.Expr,
    facets: npt.NDArray[np.int32],
) -> dolfinx.fem.Function:
    """
    Interpolate a UFL-expression into a function space, only for the degrees of freedom assoicated with facets.
    """
    domain = Q.mesh
    Q_el = Q.element
    fdim = domain.topology.dim - 1

    # Get coordinate element for facets of cell
    c_el = domain.ufl_domain().ufl_coordinate_element()
    facet_types = basix.cell.subentity_types(domain.basix_cell())[fdim]
    unique_facet_types = np.unique(facet_types)
    assert len(unique_facet_types) == 1, (
        "All facets must have the same type for interpolation."
    )
    facet_type = facet_types[0]
    x_type = domain.geometry.x.dtype
    facet_cmap = basix.ufl.element(
        "Lagrange", facet_type, c_el.degree, shape=(domain.geometry.dim,), dtype=x_type
    )
    if np.issubdtype(x_type, np.float32):
        facet_cel = dolfinx.cpp.fem.CoordinateElement_float32(
            facet_cmap.basix_element._e
        )
    elif np.issubdtype(x_type, np.float64):
        facet_cel = dolfinx.cpp.fem.CoordinateElement_float64(
            facet_cmap.basix_element._e
        )
    else:
        raise TypeError(
            f"Unsupported coordinate element type: {x_type}. "
            "Only float32 and float64 are supported."
        )
    # Pull back interpolation points from reference coordinate element to facet reference element
    ref_top = c_el.reference_topology
    ref_geom = c_el.reference_geometry
    reference_facet_points = None
    interpolation_points = Q_el.basix_element.x
    for i, points in enumerate(interpolation_points[fdim]):
        geom = ref_geom[ref_top[fdim][i]]
        ref_points = facet_cel.pull_back(points, geom)
        # Assert that interpolation points are all equal on all facets
        if reference_facet_points is None:
            reference_facet_points = ref_points
        else:
            assert np.allclose(reference_facet_points, ref_points)

    assert isinstance(reference_facet_points, np.ndarray)

    # Create expression for BC
    bndry_expr = dolfinx.fem.Expression(expr, reference_facet_points)

    # Compute number of interpolation points per sub entity
    points_per_entity = [sum(ip.shape[0] for ip in ips) for ips in interpolation_points]
    offsets = np.zeros(domain.topology.dim + 2, dtype=np.int32)
    offsets[1:] = np.cumsum(points_per_entity[: domain.topology.dim + 1])
    values_per_entity = np.zeros(
        (offsets[-1], domain.geometry.dim), dtype=dolfinx.default_scalar_type
    )

    # Map facet indices to (cell, local_facet) pairs
    boundary_entities = dolfinx.fem.compute_integration_domains(
        dolfinx.fem.IntegralType.exterior_facet, domain.topology, facets
    )

    # Compute and insert the correct values for the interpolation points on the facets
    entities = boundary_entities.reshape(-1, 2)
    values = np.zeros(entities.shape[0] * offsets[-1] * domain.geometry.dim)
    for i, entity in enumerate(entities):
        insert_pos = offsets[fdim] + reference_facet_points.shape[0] * entity[1]
        normal_on_facet = bndry_expr.eval(domain, entity.reshape(1, 2))
        values_per_entity[insert_pos : insert_pos + reference_facet_points.shape[0]] = (
            normal_on_facet.reshape(-1, domain.geometry.dim)
        )
        values[
            i * offsets[-1] * domain.geometry.dim : (i + 1)
            * offsets[-1]
            * domain.geometry.dim
        ] = values_per_entity.reshape(-1)
    # Use lower-level interpolation that takes in the function evaluated at the physical
    # interpolation points of the mesh.
    qh = dolfinx.fem.Function(Q)
    qh._cpp_object.interpolate(
        values.reshape(-1, domain.geometry.dim).T.copy(), boundary_entities[::2].copy()
    )
    qh.x.scatter_forward()
    return qh


sigma_facets = dolfinx.mesh.locate_entities_boundary(
    mesh, mesh.topology.dim - 1, Gamma_N
)
n = ufl.FacetNormal(mesh)
g = ufl.dot(ufl.grad(u_exact), n)
sigma_bc = interpolate_facet_expression(Q, g, sigma_facets)
bc_sigma = dolfinx.fem.dirichletbc(
    sigma_bc,
    dolfinx.fem.locate_dofs_topological(Q, mesh.topology.dim - 1, sigma_facets),
)
assert len(sigma_facets) == 0

# Now that we have created the bilinear and linear form, and the boundary conditions,
# we turn to solving the problem. For this we use the `dolfinx.fem.petsc.LinearProblem` class.
# As opposed to the previous examples, we now have an explicit block structure, which we would like to
# exploit when solving the problem. However, first we will solve the problem without any preconditioner
# to have a baseline performance.

# +
import dolfinx.fem.petsc

problem = dolfinx.fem.petsc.LinearProblem(
    a_blocked,
    L_blocked,
    bcs=[bc_sigma],
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "ksp_error_if_not_converged": True,
        "mat_mumps_icntl_24": 1,
        "mat_mumps_icntl_25": 0,
    },
    kind="mpi",
    petsc_options_prefix="mixed_poisson_direct",
)
# -

# Note that we have specified `kind="mpi"` in the initialization of the `LinearProblem`.
# This is to inform DOLFINx that we wan to preserve the block structure of the problem when assembling.

import time

start = time.perf_counter()
(sigma_h, u_h) = problem.solve()
end = time.perf_counter()
print(f"Direct solver took {end - start:.2f} seconds.")

L2_u = dolfinx.fem.form(ufl.inner(u_h - u_exact, u_h - u_exact) * ufl.dx)
Hdiv_sigma = dolfinx.fem.form(
    ufl.inner(
        ufl.div(sigma_h) - ufl.div(ufl.grad(u_exact)),
        ufl.div(sigma_h) - ufl.div(ufl.grad(u_exact)),
    )
    * ufl.dx
)
local_u_error = dolfinx.fem.assemble_scalar(L2_u)
local_sigma_error = dolfinx.fem.assemble_scalar(Hdiv_sigma)
u_error = np.sqrt(mesh.comm.allreduce(local_u_error, op=MPI.SUM))
sigma_error = np.sqrt(mesh.comm.allreduce(local_sigma_error, op=MPI.SUM))

print(f"Direct solver, L2(u): {u_error:.2e}, H(div)(sigma): {sigma_error:.2e}")

# ## Iterative solver with Schur complement preconditioner
# As mentioned earlier, there are more efficient ways of solving this problem, than using a direct solver.
# Especially with the saddle point structure of the problem, we can use a Schur complement preconditioner.
# As described in [FEniCSx PCTools: Mixed Poisson](https://rafinex-external-rifle.gitlab.io/fenicsx-pctools/demo/demo_mixed-poisson.html),
# Instead of wrapping the matrices in a custom wrapper, we can use `dolfinx.fem.petsc.LinearProblem` to solve the problem.

# We start by defining the $S$ matrix in the Schur complement (see the aforementioned link for details on the variational formulation).

# +
alpha = dolfinx.fem.Constant(mesh, 4.0)
gamma = dolfinx.fem.Constant(mesh, 9.0)
h = ufl.CellDiameter(mesh)
s = -(
    ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    - ufl.inner(ufl.avg(ufl.grad(v)), ufl.jump(u, n)) * ufl.dS
    - ufl.inner(ufl.jump(u, n), ufl.avg(ufl.grad(v))) * ufl.dS
    + (alpha / ufl.avg(h)) * ufl.inner(ufl.jump(u, n), ufl.jump(v, n)) * ufl.dS
    - ufl.inner(ufl.grad(u), v * n) * dGammaD
    - ufl.inner(u * n, ufl.grad(v)) * dGammaD
    + (gamma / h) * u * v * dGammaD
)

S = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(s))
S.assemble()


class SchurInv:
    def setUp(self, pc):
        self.ksp = PETSc.KSP().create(mesh.comm)
        self.ksp.setOptionsPrefix(pc.getOptionsPrefix() + "SchurInv_")
        self.ksp.setOperators(S)
        self.ksp.setTolerances(atol=1e-10, rtol=1e-10)
        self.ksp.setFromOptions()

    def apply(self, pc, x, y):
        self.ksp.solve(x, y)

    def __del__(self):
        self.ksp.destroy()


# -

# Next we can create the linear problem instance with all the required options

u_it = dolfinx.fem.Function(V, name="u_it")
sigma_it = dolfinx.fem.Function(Q, name="sigma_it")
petsc_options = {
    "ksp_error_if_not_converged": True,
    "ksp_type": "gmres",
    "ksp_rtol": 1e-10,
    "ksp_atol": 1e-10,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "upper",
    "pc_fieldsplit_schur_precondition": "user",
    f"fieldsplit_{sigma_it.name}_0_ksp_type": "preonly",
    f"fieldsplit_{sigma_it.name}_0_pc_type": "bjacobi",
    f"fieldsplit_{u_it.name}_1_ksp_type": "preonly",
    f"fieldsplit_{u_it.name}_1_pc_type": "python",
    f"fieldsplit_{u_it.name}_1_pc_python_type": __name__ + ".SchurInv",
    f"fieldsplit_{u_it.name}_1_SchurInv_ksp_type": "preonly",
    f"fieldsplit_{u_it.name}_1_SchurInv_pc_type": "hypre",
}
w_it = (sigma_it, u_it)
problem = dolfinx.fem.petsc.LinearProblem(
    a_blocked,
    L_blocked,
    u=w_it,
    bcs=[bc_sigma],
    petsc_options=petsc_options,
    petsc_options_prefix="mp_",
    kind="nest",
)

# ```{admonition} NEST matrices
# Note that instead of using `kind="mpi"` we use `kind="nest"` to indicate that we want to use a nested matrix structure
# and employ the power of [PETSc fieldsplit](https://petsc.org/release/manual/ksp/#solving-block-matrices-with-pcfieldsplit).
# ```
start_it = time.perf_counter()
problem.solve()
end_it = time.perf_counter()
print(
    f"Iterative solver took {end_it - start_it:.2f} seconds"
    + f" in {problem.solver.getIterationNumber()} iterations"
)

# We compute the error norms for the iterative solution

L2_u_it = dolfinx.fem.form(ufl.inner(u_it - u_exact, u_it - u_exact) * ufl.dx)
Hdiv_sigma_it = dolfinx.fem.form(
    ufl.inner(
        ufl.div(sigma_it) - ufl.div(ufl.grad(u_exact)),
        ufl.div(sigma_it) - ufl.div(ufl.grad(u_exact)),
    )
    * ufl.dx
)
local_u_error_it = dolfinx.fem.assemble_scalar(L2_u_it)
local_sigma_error_it = dolfinx.fem.assemble_scalar(Hdiv_sigma_it)
u_error_it = np.sqrt(mesh.comm.allreduce(local_u_error_it, op=MPI.SUM))
sigma_error_it = np.sqrt(mesh.comm.allreduce(local_sigma_error_it, op=MPI.SUM))

print(f"Iterative solver, L2(u): {u_error_it:.2e}, H(div)(sigma): {sigma_error_it:.2e}")

np.testing.assert_allclose(u_h.x.array, u_it.x.array, rtol=1e-7, atol=1e-7)
np.testing.assert_allclose(sigma_h.x.array, sigma_it.x.array, rtol=1e-7, atol=1e-7)


# ```{bibliography}
#    :filter: cited and ({"chapter4/mixed_poisson"} >= docnames)
# ```
