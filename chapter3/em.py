# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Electromagnetics example
#
# Theoretical introduction by: Hans Petter Langtangen and Anders Logg
#
# Implementation by: Jørgen S. Dokken
#
# In this example, we will consider an iron cylinder with copper wires wound around the cylinder, as shown below
#
# ![Cross section of wires](wire.png)
#
# Through the copper wires a static current is flowing, with current density $+1\,\mathrm{A/m^2}$ in the inner wires
# and $-1\,\mathrm{A/m^2}$ in the outer wires.
# We would like to compute the magnetic field $B$ in the iron cylinder, the copper wires, and the surrounding vacuum.
#
# We will reduce this to a problem on the 2D cross section of the geometry,
# and derive a Poisson equation for the magnetic field (or rather its potential).
#
# ## Reduction of Maxwell's equations to a 2D Poisson problem
#
# We start with Maxwell's equations
#
# $$
# \nabla \cdot D = \rho,
# $$
# $$
# \nabla \cdot B = 0,
# $$
# $$
# \nabla \times E = -\frac{\partial B}{\partial t},
# $$
# $$
# \nabla \times H = \frac{\partial D}{\partial t}+ J.
# $$
#
# Here, $D$ is the displacement field, $B$ is the magnetic field, $E$ is the electric field, and $H$ is the magnetizing field.
# In addition to Maxwell's equations, we need a constitutive relation between $B$ and $H$,
#
# $$
# B =\mu H,
# $$
#
# which holds for an isotropic linear magnetic medium.
# Here, $\mu$ is the magnetic permeability of the material.
# Since $B$ is solenoidal (divergence free) according to Maxwell's equations, $B$ must be the curl of some vector field $A$ in $\mathbb{R}^3$.
# This field is called the magnetic vector potential. Since the problem is static (assumption 3 below) and thus $\frac{\partial D}{\partial t}=0$, it follows that
#
# $$
# J = \nabla \times H = \nabla \times(\mu^{-1} B)=\nabla \times (\mu^{-1}\nabla \times A ).
# $$
#
# So far, apart from the problem being static, everything holds for a general 3D problem.
# Note that $\mu$ varies in space.
# It is piecewise constant, with jumps at all material interfaces.
# The largest jump is between the iron cylinder and its surroundings,
# as the permeabilities of copper and vacuum are almost equal.
# We can therefore not use the identity $\nabla \times (\nabla \times A) = \nabla(\nabla\cdot A) - \nabla^2 A$ directly,
# as $\mu^{-1}$ cannot be moved outside the outer curl.
# Instead, we use the 2D structure of the problem.
#
# ### Assumptions of the 2D model
#
# We make the following assumptions:
#
# 1. The cylinder and the wires extend far along the $z$-axis, so that all fields are independent of the $z$-coordinate.
# 2. The current only flows in the $z$-direction, $J=(0,0,J_z(x,y))$.
# 3. The problem is static, $\frac{\partial D}{\partial t}=0$.
# 4. The materials are linear and isotropic, with a piecewise constant permeability $\mu=\mu(x,y)$.
# 5. There are no surface currents at the material interfaces.
# 6. The magnetic field and its potential vanish far away from the wires.
#    The magnetic field decays for any net current through the cross section,
#    while the potential can only vanish if the net current is zero (otherwise it grows logarithmically).
#    This is the case here, as the currents in the inner and outer wires cancel.
#
# By assumption 3, $\nabla \times H = J$.
# As all derivatives with respect to $z$ vanish by assumption 1, and $J=(0,0,J_z)$ by assumption 2, this equation reads
#
# $$
# \nabla \times H = \left(\frac{\partial H_z}{\partial y}, -\frac{\partial H_z}{\partial x}, \frac{\partial H_y}{\partial x} - \frac{\partial H_x}{\partial y}\right) = (0, 0, J_z).
# $$
#
# The $z$-component will give us the Poisson equation for the magnetic vector potential below.
# The $x$- and $y$-components give $\frac{\partial H_z}{\partial y} = \frac{\partial H_z}{\partial x} = 0$.
# Thus $H_z$ is constant within each connected component of the material subdomains
# (the vacuum, for instance, consists of the hole inside the iron cylinder and the exterior).
# As $H_z$ is tangential to the material interfaces, it is continuous across them by assumption 5,
# and therefore constant in the whole plane.
# By assumption 6, $H_z=B_z=0$.
# In other words, $B=B(x,y)$ only has $x$- and $y$-components.
#
# ### Choice of gauge
#
# The vector potential $A$ is not unique; if $B=\nabla \times A$, then $B=\nabla \times (A+\nabla \phi)$ for any scalar field $\phi$ (the gauge function),
# as $\nabla \times \nabla \phi = 0$.
# Conversely, on a simply connected domain such as $\mathbb{R}^3$, the difference of two vector potentials of $B$ is curl free,
# and thus they differ by such a gradient.
# A gauge is an additional condition imposed on $A$, which restricts this freedom.
#
# As $\frac{\partial B_x}{\partial x} + \frac{\partial B_y}{\partial y} = \nabla \cdot B = 0$, there exists a scalar function $\psi(x,y)$ (a stream function)
# such that $B = \left(\frac{\partial \psi}{\partial y}, -\frac{\partial \psi}{\partial x}, 0\right)$.
# Thus, $A=(0,0,\psi)$ is a vector potential of $B$.
# We use this as our gauge condition, i.e. we require that $A$ is of the form $A=(0,0,A_z(x,y))$.
# This choice also satisfies the Coulomb gauge, $\nabla \cdot A = \frac{\partial A_z}{\partial z}=0$.
# However, it does not determine $A$ completely.
# The gauge function $\phi = cz$, for any constant $c$, gives $\nabla \phi = (0,0,c)$,
# so $A_z$ and $A_z + c$ give the same magnetic field $B$.
# We remove this freedom through the boundary condition below.
#
# ### The scalar Poisson equation
#
# With $A=(0,0,A_z(x,y))$ we have
#
# $$
# B = \nabla \times A = \left(\frac{\partial A_z}{\partial y}, -\frac{\partial A_z}{\partial x}, 0\right).
# $$
#
# As $\mu^{-1}B$ is also independent of $z$ and has no $z$-component, the $x$- and $y$-components of its curl vanish, while the $z$-component is
#
# $$
# \left(\nabla \times (\mu^{-1}B)\right)_z = \frac{\partial}{\partial x}\left(-\mu^{-1}\frac{\partial A_z}{\partial x}\right) - \frac{\partial}{\partial y}\left(\mu^{-1}\frac{\partial A_z}{\partial y}\right) = -\nabla \cdot (\mu^{-1}\nabla A_z).
# $$
#
# Equating the $z$-components of $J$ and $\nabla \times (\mu^{-1}\nabla \times A)$ gives $-\nabla \cdot (\mu^{-1}\nabla A_z) = J_z$, without any assumption of $\mu$ being constant.
# As $\mu$ is discontinuous, this equation should be understood in the weak sense.
# At a material interface with unit normal $n=(n_x, n_y)$ and unit tangent $t=(-n_y, n_x)$,
# the normal component of $B$ and, by assumption 5, the tangential component of $H$ are continuous.
# Inserting the expression for $B$, we obtain
#
# $$
# B\cdot n = \frac{\partial A_z}{\partial t}, \qquad H \cdot t = -\mu^{-1}\frac{\partial A_z}{\partial n}.
# $$
#
# Continuity of $B\cdot n$ only implies that the tangential derivative of $A_z$ is continuous,
# i.e. that $A_z$ is continuous up to a constant along each interface.
# A jump in $A_z$ would however give a singular contribution to $B$ on the interface, and an infinite magnetic energy.
# We therefore require that $A_z\in H^1_{\mathrm{loc}}(\mathbb{R}^2)$, i.e. that $A_z$ is in $H^1$ on every bounded subset of $\mathbb{R}^2$, which rules out such jumps.
# Thus, $A_z$ and $\mu^{-1}\frac{\partial A_z}{\partial n}$ should be continuous across material interfaces.
#
# ### Boundary conditions and uniqueness
#
# For the current problem, we thus need to solve the following 2D Poisson problem for the $z$-component $A_z$ of the magnetic vector potential
#
# $$
#     - \nabla \cdot (\mu^{-1} \nabla A_z) = J_z \qquad \text{in } \mathbb{R}^2,\\
# $$
# $$
# \lim_{\vert(x,y)\vert\to \infty}A_z = 0.
# $$
#
# The condition at infinity is what makes the solution unique.
# Without it, we could add any function $w$ satisfying $-\nabla \cdot (\mu^{-1}\nabla w) = 0$ to $A_z$.
# This includes the constant shift $w=c$ from the gauge freedom above, which does not change $B$.
# It also includes non-constant functions that do change $B$, for instance $w=ax+by$ if $\mu$ were constant,
# which adds the uniform background field $(b, -a, 0)$ to $B$.
# Requiring that $A_z$ vanishes at infinity rules out both.
# Without any boundary condition, these additional solutions form an infinite-dimensional space,
# so we cannot remove them by attaching a nullspace to the linear system.
#
# Since we cannot solve the problem on an infinite domain, we will truncate the domain using a large disk $\Omega$, and set $A_z=0$ on its boundary $\partial\Omega$.
# This Dirichlet condition approximates the condition at infinity, and makes the linear system non-singular.
# Alternatively, we could use the natural (homogeneous Neumann) condition $\mu^{-1}\frac{\partial A_z}{\partial n}=0$, i.e. $H\cdot t=0$, on $\partial\Omega$.
# Then, only the constants remain as additional solutions, which could be handled
# by attaching the nullspace as in the [singular Poisson problem](../chapter2/singular_poisson).
# This requires the net current to be zero, i.e. the compatibility condition $\int_\Omega J_z~\mathrm{d}x = 0$.
# For the discretized problem, this condition only holds up to the geometry approximation error,
# as the area of each polygonal wire cross section differs slightly from $\pi r^2$.
# The two conditions give slightly different solutions on a finite disk.
# With $A_z=0$, the magnetic field is tangential to $\partial\Omega$ ($B\cdot n=0$),
# while with $H\cdot t=0$ it crosses $\partial\Omega$ at a right angle.
# Both converge to the solution in $\mathbb{R}^2$ as the radius of the disk increases.
#
# The current density $J_z$ is set to $+1\,\mathrm{A/m^2}$ in the interior set of the circles (copper-wire cross sections) and to $-1\,\mathrm{A/m^2}$ in the exterior set of circles in the cross section figure.
#
# ### Weak formulation
#
# We multiply the Poisson equation by a test function $v$ that vanishes on $\partial\Omega$,
# and integrate by parts over each material subdomain $\Omega_i$ separately, as $\mu$ is discontinuous across their boundaries.
# This gives
#
# $$
# \sum_i \int_{\Omega_i} \mu^{-1}\nabla A_z \cdot \nabla v ~\mathrm{d}x - \sum_i \int_{\partial\Omega_i} \mu^{-1}\frac{\partial A_z}{\partial n_i} v ~\mathrm{d}s = \int_\Omega J_z v~\mathrm{d} x,
# $$
#
# where $n_i$ is the outward pointing unit normal of $\Omega_i$.
# The integrals over $\partial\Omega$ vanish, as $v=0$ there.
# Each interface $\Gamma$ between two subdomains appears twice, with opposite normals,
# and the two contributions combine to the integral of the jump of $\mu^{-1}\frac{\partial A_z}{\partial n}$ across $\Gamma$, multiplied by $v$.
# By continuity of $H\cdot t$, these interface terms vanish.
# Conversely, any $A_z$ satisfying the resulting weak formulation for all $v$ has a continuous $\mu^{-1}\frac{\partial A_z}{\partial n}$ across interfaces.
# In other words, this is a natural condition of the weak formulation, and it is therefore only satisfied weakly by the discrete solution.
# Continuity of $A_z$ (and thereby of $B\cdot n$) is on the other hand imposed by choosing a continuous (Lagrange) function space for $A_z$.
#
# We obtain $a(A_z,v)=L(v)$ with
#
# $$
# a(A_z, v)=\int_\Omega \mu^{-1}\nabla A_z \cdot \nabla v ~\mathrm{d}x,
# $$
# $$
# L(v)=\int_\Omega J_z v~\mathrm{d} x.
# $$
#
# Once the magnetic vector potential has been computed, we compute the magnetic field from $B=\nabla \times A$ derived above,
# which for the 2D problem reads
#
# $$
#     B(x,y)=\left(\frac{\partial A_z}{\partial y}, - \frac{\partial A_z}{\partial x} \right).
# $$

# ## Meshing a complex structure with subdomains
#
# We create the domain visualized in the cross section figure above using gmsh. Note that we are using the `gmsh.model.occ.fragment` commands to ensure that the boundaries of the wires are resolved in the mesh.

# +
from dolfinx import default_scalar_type
from dolfinx.fem import (
    dirichletbc,
    Expression,
    Function,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.io import gmsh as gmshio
from dolfinx.mesh import compute_midpoints, locate_entities_boundary
from dolfinx.plot import vtk_mesh

from ufl import TestFunction, TrialFunction, as_vector, dot, dx, grad
from mpi4py import MPI

import gmsh
import numpy as np
import pyvista

rank = MPI.COMM_WORLD.rank

gmsh.initialize()
r = 0.1  # Radius of copper wires
R = 5  # Radius of domain
a = 1  # Radius of inner iron cylinder
b = 1.2  # Radius of outer iron cylinder
N = 8  # Number of windings
c_1 = 0.8  # Radius of inner copper wires
c_2 = 1.4  # Radius of outer copper wires
gdim = 2  # Geometric dimension of the mesh
model_rank = 0
mesh_comm = MPI.COMM_WORLD
if mesh_comm.rank == model_rank:
    # Define geometry for iron cylinder
    outer_iron = gmsh.model.occ.addCircle(0, 0, 0, b)
    inner_iron = gmsh.model.occ.addCircle(0, 0, 0, a)
    gmsh.model.occ.addCurveLoop([outer_iron], 5)
    gmsh.model.occ.addCurveLoop([inner_iron], 6)
    iron = gmsh.model.occ.addPlaneSurface([5, 6])
    gmsh.model.occ.synchronize()

    # Define geometry for background
    background = gmsh.model.occ.addDisk(0, 0, 0, R, R)
    gmsh.model.occ.synchronize()

    # Define the copper-wires inside iron cylinder
    angles_N = [i * 2 * np.pi / N for i in range(N)]
    wires_N = [
        (2, gmsh.model.occ.addDisk(c_1 * np.cos(v), c_1 * np.sin(v), 0, r, r))
        for v in angles_N
    ]

    # Define the copper-wires outside the iron cylinder
    angles_S = [(i + 0.5) * 2 * np.pi / N for i in range(N)]
    wires_S = [
        (2, gmsh.model.occ.addDisk(c_2 * np.cos(v), c_2 * np.sin(v), 0, r, r))
        for v in angles_S
    ]
    gmsh.model.occ.synchronize()
    # Resolve all boundaries of the different wires in the background domain
    all_surfaces = [(2, iron)]
    all_surfaces.extend(wires_S)
    all_surfaces.extend(wires_N)
    whole_domain = gmsh.model.occ.fragment([(2, background)], all_surfaces)
    gmsh.model.occ.synchronize()
    # Create physical markers for the different wires.
    # We use the following markers:
    # - Vacuum: 0
    # - Iron cylinder: 1
    # - Inner copper wires: $[2,3,\dots,N+1]$
    # - Outer copper wires: $[N+2,\dots, 2\cdot N+1]
    inner_tag = 2
    outer_tag = 2 + N
    background_surfaces = []
    other_surfaces = []
    for domain in whole_domain[0]:
        com = gmsh.model.occ.getCenterOfMass(domain[0], domain[1])
        mass = gmsh.model.occ.getMass(domain[0], domain[1])
        # Identify iron circle by its mass
        if np.isclose(mass, np.pi * (b**2 - a**2)):
            gmsh.model.addPhysicalGroup(domain[0], [domain[1]], tag=1)
            other_surfaces.append(domain)
        # Identify the background circle by its center of mass
        elif np.allclose(com, [0, 0, 0]):
            background_surfaces.append(domain[1])

        # Identify the inner circles by their center of mass
        elif np.isclose(np.linalg.norm(com), c_1):
            gmsh.model.addPhysicalGroup(domain[0], [domain[1]], inner_tag)
            inner_tag += 1
            other_surfaces.append(domain)
        # Identify the outer circles by their center of mass
        elif np.isclose(np.linalg.norm(com), c_2):
            gmsh.model.addPhysicalGroup(domain[0], [domain[1]], outer_tag)
            outer_tag += 1
            other_surfaces.append(domain)
    # Add marker for the vacuum
    gmsh.model.addPhysicalGroup(2, background_surfaces, tag=0)
    # Create mesh resolution that is fine around the wires and
    # iron cylinder, coarser the further away you get
    gmsh.model.mesh.field.add("Distance", 1)
    edges = gmsh.model.getBoundary(other_surfaces, oriented=False)
    gmsh.model.mesh.field.setNumbers(1, "EdgesList", [e[1] for e in edges])
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "IField", 1)
    gmsh.model.mesh.field.setNumber(2, "LcMin", r / 3)
    gmsh.model.mesh.field.setNumber(2, "LcMax", 6 * r)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 4 * r)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 10 * r)
    gmsh.model.mesh.field.add("Min", 5)
    gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2])
    gmsh.model.mesh.field.setAsBackgroundMesh(5)
    # Generate mesh
    gmsh.option.setNumber("Mesh.Algorithm", 7)
    gmsh.model.mesh.generate(gdim)
    gmsh.model.mesh.optimize("Netgen")
# -

# As in [the Navier-Stokes tutorial](../chapter2/ns_code2) we load the mesh directly into DOLFINx, without writing it to file.

mesh_data = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=2)
mesh = mesh_data.mesh
assert mesh_data.cell_tags is not None
ct = mesh_data.cell_tags
gmsh.finalize()

# To inspect the mesh, we use Paraview, and obtain the following mesh

with XDMFFile(MPI.COMM_WORLD, "mt.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(ct, mesh.geometry)

# We can also visualize the subdommains using pyvista

# Each process builds the grid of the cells it owns, the pieces are gathered on one process and
# merged into a single grid, which is then drawn, see
# [Plotting in parallel](../chapter1/fundamentals_code).

tdim = mesh.topology.dim
mesh.topology.create_connectivity(tdim, tdim)
grid = pyvista.UnstructuredGrid(*vtk_mesh(mesh, tdim))
num_local_cells = mesh.topology.index_map(tdim).size_local
markers = ct.values[ct.indices < num_local_cells]
grid.cell_data["Marker"] = markers
grid.set_active_scalars("Marker")

root = 0
assert root < mesh.comm.size, f"Cannot gather on process {root} of {mesh.comm.size}"

marker_clim = [
    mesh.comm.reduce(markers.min(), op=MPI.MIN, root=root),
    mesh.comm.reduce(markers.max(), op=MPI.MAX, root=root),
]
marker_pieces = mesh.comm.gather(grid, root=root)

if marker_pieces is not None:
    marker_grid = pyvista.merge(marker_pieces)
    plotter = pyvista.Plotter()
    actor = plotter.add_mesh(marker_grid, show_edges=True, clim=marker_clim)
    plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        cell_tag_fig = plotter.screenshot("cell_tags.png")


# Next, we define the discontinuous functions for the permeability $\mu$ and current $J_z$ using the `MeshTags` as in [Defining material parameters through subdomains](./subdomains)
#

Q = functionspace(mesh, ("DG", 0))
material_tags = np.unique(ct.values)
mu = Function(Q)
J = Function(Q)
# As we only set some values in J, initialize all as 0
J.x.array[:] = 0
for tag in material_tags:
    cells = ct.find(tag)
    # Set values for mu
    if tag == 0:
        mu_ = 4 * np.pi * 1e-7  # Vacuum
    elif tag == 1:
        mu_ = 1e-5  # Iron (This should really be 6.3e-3)
    else:
        mu_ = 1.26e-6  # Copper
    mu.x.array[cells] = np.full_like(cells, mu_, dtype=default_scalar_type)
    if tag in range(2, 2 + N):
        J.x.array[cells] = np.full_like(cells, 1, dtype=default_scalar_type)
    elif tag in range(2 + N, 2 * N + 2):
        J.x.array[cells] = np.full_like(cells, -1, dtype=default_scalar_type)

# In the code above, we have used a somewhat less extreme value for the magnetic permeability of iron. This is to make the solution a little more interesting. It would otherwise be completely dominated by the field in the iron cylinder.
#
# We can now define the weak problem

# +
V = functionspace(mesh, ("Lagrange", 1))
tdim = mesh.topology.dim
facets = locate_entities_boundary(mesh, tdim - 1, lambda x: np.full(x.shape[1], True))
dofs = locate_dofs_topological(V, tdim - 1, facets)
bc = dirichletbc(default_scalar_type(0), dofs, V)

u = TrialFunction(V)
v = TestFunction(V)
a = (1 / mu) * dot(grad(u), grad(v)) * dx
L = J * v * dx
# -

# We are now ready to solve the linear problem

A_z = Function(V)
problem = LinearProblem(a, L, u=A_z, bcs=[bc], petsc_options_prefix="em_")
problem.solve()

# As we have computed the magnetic potential, we can now compute the magnetic field, by setting `B=curl(A_z)`. Note that as we have chosen a function space of first order piecewise linear function to describe our potential, the curl of a function in this space is a discontinuous zeroth order function (a function of cell-wise constants). We use `dolfinx.fem.Expression` to interpolate the curl into `W`.

W = functionspace(mesh, ("DG", 0, (mesh.geometry.dim,)))
B = Function(W)
B_expr = Expression(as_vector((A_z.dx(1), -A_z.dx(0))), W.element.interpolation_points)
B.interpolate(B_expr)

# Note that we used `ufl.as_vector` to interpret the `Python`-tuple `(A_z.dx(1), -A_z.dx(0))` as a vector in the unified form language (UFL).
#
# We now plot the magnetic potential $A_z$ and the magnetic field $B$. We start by creating a new plotter

# +
Az_grid = pyvista.UnstructuredGrid(*vtk_mesh(V))
Az_grid.point_data["A_z"] = A_z.x.array
Az_grid.set_active_scalars("A_z")

Az_clim = [
    mesh.comm.reduce(A_z.x.array.min(), op=MPI.MIN, root=root),
    mesh.comm.reduce(A_z.x.array.max(), op=MPI.MAX, root=root),
]
Az_pieces = mesh.comm.gather(Az_grid, root=root)

if Az_pieces is not None:
    merged_Az_grid = pyvista.merge(Az_pieces)
    plotter = pyvista.Plotter()
    actor = plotter.add_mesh(
        merged_Az_grid.warp_by_scalar("A_z", factor=1e7), show_edges=True, clim=Az_clim
    )
    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        Az_fig = plotter.screenshot("Az.png")
# -

# ## Visualizing the magnetic field
# As the magnetic field is a piecewise constant vector field, we need create a custom plotting function.
# We start by computing the midpoints of each cell, which is where we would like to visualize the cell-wise constant vector.
# Next, we take the data from the function `B`, and  shape it to become a 3D vector.
# We connect the vector field with the midpoint by using `pyvista.PolyData`.

# +
# We use the cells owned by the process, so that no arrow is drawn twice where two processes
# meet, and gather the clouds along with the grid of the mesh
top_imap = mesh.topology.index_map(mesh.topology.dim)
num_cells = top_imap.size_local
mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)
midpoints = compute_midpoints(
    mesh, mesh.topology.dim, np.arange(num_cells, dtype=np.int32)
)

num_dofs = W.dofmap.index_map.size_local
assert num_cells == num_dofs
bs = W.dofmap.index_map_bs
values = np.zeros((num_dofs, 3), dtype=np.float64)
values[:, : mesh.geometry.dim] = B.x.array.real[: num_dofs * bs].reshape(num_dofs, bs)
cloud = pyvista.PolyData(midpoints)
cloud["B"] = values

cloud_pieces = mesh.comm.gather(cloud, root=root)
mesh_pieces = mesh.comm.gather(grid, root=root)

if cloud_pieces is not None and mesh_pieces is not None:
    merged_cloud = pyvista.merge(cloud_pieces)
    merged_grid = pyvista.merge(mesh_pieces)
    plotter = pyvista.Plotter()
    plotter.set_position([0, 0, 5])
    actor = plotter.add_mesh(merged_grid, style="wireframe", color="k")
    actor2 = plotter.add_mesh(merged_cloud.glyph("B", factor=2e6))

    if not pyvista.OFF_SCREEN:
        plotter.show(screenshot="B.png")
    else:
        B_fig = plotter.screenshot("B.png")
