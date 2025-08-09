# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.2
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
import dolfinx

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 50, 50)
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
# right-hand side $\tilde f$.
# \begin{align}
# \begin{pmatrix}
# A & B^T\\
# B & 0
# \end{pmatrix}
# \begin{pmatrix}
# u_h\\
# sigma_h
# \end{pmatrix}
# = \begin{pmatrix}
# 0\\
# \tilde f
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
# For `sigma`, we have a manufactured solution that depends on the normal vector on the boundary,
# which makes it slightly more complicated to implement.


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

(sigma_h, u_h) = problem.solve()

L2_u = dolfinx.fem.form(ufl.inner(u_h - u_exact, u_h - u_exact) * ufl.dx)
L2_sigma = dolfinx.fem.form(
    ufl.inner(sigma_h - ufl.grad(u_exact), sigma_h - ufl.grad(u_exact)) * ufl.dx
)
local_u_error = dolfinx.fem.assemble_scalar(L2_u)
local_sigma_error = dolfinx.fem.assemble_scalar(L2_sigma)
u_error = np.sqrt(mesh.comm.allreduce(local_u_error, op=MPI.SUM))
sigma_error = np.sqrt(mesh.comm.allreduce(local_sigma_error, op=MPI.SUM))


print(f"u error: {u_error:.3e}")
print(f"sigma error: {sigma_error:.3e}")
u_D.name = "u_ex"
with dolfinx.io.XDMFFile(mesh.comm, "mixed_poisson.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_h, 0.0)
    xdmf.write_function(u_D, 0.0)

# ```{bibliography}
#    :filter: cited and ({"chapter4/mixed_poisson"} >= docnames)
# ```
