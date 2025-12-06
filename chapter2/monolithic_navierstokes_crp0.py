from __future__ import annotations
import argparse
import numpy as np
from mpi4py import MPI
from dolfinx import fem, io, mesh
from dolfinx.fem import petsc
from petsc4py import PETSc
import ufl
from basix.ufl import element, mixed_element


def parse_args():
    p = argparse.ArgumentParser(description="CR–P0 steady Navier–Stokes with Nitsche BCs (robust)")
    p.add_argument("--nx", type=int, default=64)
    p.add_argument("--ny", type=int, default=32)
    p.add_argument("--Lx", type=float, default=2.0)
    p.add_argument("--Ly", type=float, default=1.0)
    p.add_argument("--nu", type=float, default=1e-2)
    p.add_argument("--uin", type=float, default=1.0)       
    p.add_argument("--picard-it", type=int, default=20)
    p.add_argument("--picard-tol", type=float, default=1e-8)
    p.add_argument("--theta", type=float, default=0.5)
    p.add_argument("--eps-p", type=float, default=1e-12)   # tiny pressure mass to remove nullspace
    p.add_argument("--stokes-only", action="store_true")
    p.add_argument("--no-output", action="store_true")
    p.add_argument("--alpha", type=float, default=400.0, help="Nitsche penalty (100–1000 recommended)")
    p.add_argument("--outfile", type=str, default="chapter2/open_cavity_crp0.xdmf")
    return p.parse_args()


args = parse_args()
comm = MPI.COMM_WORLD
rank = comm.rank

# --- mesh ---
domain = mesh.create_rectangle(
    comm,
    [np.array([0.0, 0.0]), np.array([args.Lx, args.Ly])],
    [args.nx, args.ny],
    cell_type=mesh.CellType.triangle,
)
tdim = domain.topology.dim
fdim = tdim - 1
gdim = domain.geometry.dim
cell = domain.basix_cell()

# Ensure needed connectivities
domain.topology.create_connectivity(fdim, tdim)
domain.topology.create_connectivity(tdim, tdim)

# --- boundary facet tags (needed for ds with Nitsche) ---
left_facets   = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0.0))
right_facets  = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], args.Lx))
bottom_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 0.0))
top_facets    = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], args.Ly))

# Tag: 1=left, 2=right, 3=bottom, 4=top
facet_indices = np.concatenate([left_facets, right_facets, bottom_facets, top_facets]).astype(np.int32)
facet_tags    = np.concatenate([
    np.full_like(left_facets,   1, dtype=np.int32),
    np.full_like(right_facets,  2, dtype=np.int32),
    np.full_like(bottom_facets, 3, dtype=np.int32),
    np.full_like(top_facets,    4, dtype=np.int32),
])

# Create facet meshtags
if facet_indices.size == 0:
    ft = mesh.meshtags(domain, fdim, np.array([], dtype=np.int32), np.array([], dtype=np.int32))
else:
    order = np.argsort(facet_indices)
    ft = mesh.meshtags(domain, fdim, facet_indices[order], facet_tags[order])

dx = ufl.Measure("dx", domain=domain)
ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)

# --- spaces: CR–P0 (mixed) ---
V_el = element("CR", cell, 1, shape=(gdim,))
Q_el = element("DG", cell, 0)
W = fem.functionspace(domain, mixed_element([V_el, Q_el]))
(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)

# --- parameters / RHS ---
nu = fem.Constant(domain, PETSc.ScalarType(args.nu))
eps_p = fem.Constant(domain, PETSc.ScalarType(args.eps_p))
zero = fem.Constant(domain, PETSc.ScalarType(0.0))
f_vec = ufl.as_vector((zero, zero))  # zero body force

# --- Picard state (for convection) ---
w = fem.Function(W)        # holds (u_k, p_k)
u_k, p_k = w.split()

def build_forms():
    n = ufl.FacetNormal(domain)
    h = ufl.CellDiameter(domain)
    alpha = PETSc.ScalarType(args.alpha)

    u_in = ufl.as_vector((PETSc.ScalarType(args.uin), PETSc.ScalarType(0.0)))
    zero_vec = ufl.as_vector((PETSc.ScalarType(0.0),  PETSc.ScalarType(0.0)))

    # Core Stokes + tiny pressure mass
    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
        - ufl.div(v) * p * dx
        + q * ufl.div(u) * dx
        + eps_p * p * q * dx
        - ufl.inner(f_vec, v) * dx
    )

    # Add convection if not stokes-only
    if not args.stokes_only:
        F += ufl.inner(ufl.dot(u_k, ufl.nabla_grad(u)), v) * dx

    # -------- Nitsche + boundary consistency terms on Dirichlet parts --------
    # Left inlet (tag=1): u = u_in
    F += (
        - nu * ufl.inner(ufl.grad(u) * n, v)                    # consistency
        - nu * ufl.inner(ufl.grad(v) * n, (u - u_in))           # symmetry
        + alpha * nu / h * ufl.inner(u - u_in, v)               # penalty
        - ufl.inner(p * n, v)                                   # momentum BC consistency
        + q * ufl.inner(u - u_in, n)                            # continuity BC consistency: q * ((u-u_in)·n)
    ) * ds(1)

    # Bottom (3) and Top (4): no-slip u = 0
    F += (
        - nu * ufl.inner(ufl.grad(u) * n, v)
        - nu * ufl.inner(ufl.grad(v) * n, (u - zero_vec))
        + alpha * nu / h * ufl.inner(u - zero_vec, v)
        - ufl.inner(p * n, v)
        + q * ufl.inner(u - zero_vec, n)
    ) * ds((3,4))

    # Right (2) left as natural outlet (do-nothing)
    return ufl.lhs(F), ufl.rhs(F)


def solve_once():
    a, L = build_forms()
    # No strong BCs with CR → Nitsche handles boundaries
    A = petsc.assemble_matrix(fem.form(a), bcs=[])
    A.assemble()
    b = A.createVecRight(); b.set(0.0)
    petsc.assemble_vector(b, fem.form(L))
    # (No lifting/BC since bcs=[])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    if rank == 0:
        PETSc.Sys.Print(f"  ||b||_2 = {b.norm():.3e}")

    # Robust Krylov (no factorization)
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("gmres")
    ksp.setTolerances(rtol=1e-8, max_it=1000)
    ksp.getPC().setType("jacobi")

    # wrap dolfinx vector, solve, copy back
    x = PETSc.Vec().createWithArray(w.x.array, comm=comm)
    ksp.solve(b, x)
    w.x.array[:] = x.getArray()
    w.x.scatter_forward()
    if rank == 0:
        PETSc.Sys.Print(f"  ||x||_2 = {x.norm():.3e}")


# --- Picard loop ---
theta = float(args.theta)
tol = float(args.picard_tol)
max_it = int(args.picard_it)

# for residual check (velocity only)
V_sub = W.sub(0)
V0, _ = V_sub.collapse()
u_prev_fun = fem.Function(V0)
u_prev_arr = np.zeros_like(u_prev_fun.x.array)

for it in range(1, max_it + 1):
    solve_once()

    # velocity from mixed solution
    u_view = w.sub(0).collapse()
    u_curr = u_view[0] if isinstance(u_view, tuple) else u_view
    u_curr_arr = u_curr.x.array.copy()

    diff = u_curr_arr - u_prev_arr
    err = float(np.linalg.norm(diff)) if np.all(np.isfinite(diff)) else float("inf")

    if rank == 0:
        PETSc.Sys.Print(f"Picard {it:02d}: ||u - u_prev|| = {err:.3e}")
        PETSc.Sys.Print(f"  |u| range: [{np.abs(u_curr_arr).min():.3e}, {np.abs(u_curr_arr).max():.3e}]")

    if not np.isfinite(err) or err < tol or args.stokes_only:
        break

    # under-relax linearization state u_k := θ u_curr + (1-θ) u_prev
    relaxed = theta * u_curr_arr + (1.0 - theta) * u_prev_arr
    u_k.x.array[:len(relaxed)] = relaxed
    u_prev_arr = u_curr_arr.copy()

# --- output ---
if not args.no_output:
    u_view = w.sub(0).collapse()
    p_view = w.sub(1).collapse()
    u_fun = u_view[0] if isinstance(u_view, tuple) else u_view
    p_fun = p_view[0] if isinstance(p_view, tuple) else p_view

    # interpolate to CG1 for cleaner XDMF viewing
  Vout = fem.functionspace(domain, element("DG", cell, 1, shape=(gdim,)))
    Qout = fem.functionspace(domain, element("Lagrange", cell, 1))
    u_out = fem.Function(Vout, name="u")
    u_out.interpolate(u_fun)

    with io.VTXWriter(domain.comm, args.outfile, [u_out, p_fun]) as writer:
        writer.write(0.0)
    if rank == 0:
        PETSc.Sys.Print(f"Wrote {args.outfile}")
