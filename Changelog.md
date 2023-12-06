# Changelog

## v0.7.2
 - Change pyvista backend to `html`, using Pyvista main branch
 - Using DOLFINx v0.7.2 https://github.com/FEniCS/dolfinx/releases/tag/v0.7.2 as base

## v0.7.1
 - No API changes, release due to various bug-fixes from the 0.7.0 release, see:
   https://github.com/FEniCS/dolfinx/releases/tag/v0.7.1 for more information

## v0.7.0

- Renamed `dolfinx.graph.create_adjacencylist` to `dolfinx.graph.adjacencylist`
- Renamed `dolfinx.plot.create_vtk_mesh` to `dolfinx.plot.vtk_mesh`
- `dolfinx.geometry.BoundingBoxTree` has been changed to `dolfinx.geometry.bb_tree`
- `create_mesh` with Meshio has been modified. Note that you now need to pass dtype `np.int32` to the cell_data.
- Update dolfinx petsc API. Now one needs to explicitly import `dolfinx.fem.petsc` and `dolfinx.fem.nls`, as PETSc is no longer a strict requirement. Replace `petsc4py.PETSc.ScalarType` with `dolfinx.default_scalar_type` in demos where we do not use `petsc4py` explicitly.

## v0.6.0

- Remove `ipygany` and `pythreejs` as plotting backends. Using `panel`.
- Add gif-output to [chapter2/diffusion_code] and [chapter2/hyperelasticity].
- Replace `dolfinx.fem.Function.geometric_dimension` with `len(dolfinx.fem.Function)`
- Improve [chapter2/ns_code2] to have better splitting scheme and density.
- Improve mesh quality in [chapter3/em].
- `jit_params` and `form_compiler_params` renamed to `*_options`.

## v0.5.0

- Using new GMSH interface in DOLFINx (`dolfinx.io.gmshio`) in all demos using GMSH
- Added a section on custom Newton-solvers, see [chapter4/newton-solver].
- Various minor DOLFINx API updates. `dolfinx.mesh.compute_boundary_facets` -> `dolfinx.mesh.exterior_facet_indices` with slightly different functionality. Use `dolfinx.mesh.MeshTagsMetaClass.find` instead of `mt.indices[mt.values==value]`.
- Various numpy updates, use `np.full_like`.
- Change all notebooks to use [jupytext](https://jupytext.readthedocs.io/en/latest/install.html) to automatically sync `.ipynb` with `.py` files.
- Add example of how to use `DOLFINx` in complex mode, see [chapter1/complex_mode].

## 0.4.1

- No changes

## 0.4.0 (05.02.2021)

- All `pyvista` plotting has been rewritten to use `ipygany` and `pythreejs` as well as using a cleaner interface.
- `dolfinx.plot.create_vtk_topology` has been renamed to `dolfinx.plot.create_vtk_mesh` and can now be directly used as input to `pyvista.UnstructuredGrid`.
- `dolfinx.fem.Function.compute_point_values` has been deprecated. Interpolation into a CG-1 is now the way of getting vertex values.
- API updates wrt. DOLFINx. `Form`->`form`, `DirichletBC`->`dirichletbc`.
- Updates on error computations in [Error control: Computing convergence rates](chapter4/convergence).
- Added tutorial on interpolation of `ufl.Expression` in [Deflection of a membrane](chapter1/membrane_code).
- Added tutorial on how to apply constant-valued Dirichet conditions in [Deflection of a membrane](chapter1/membrane_code).
- Various API changes relating to the import structure of DOLFINx

## 0.3.0 (09.09.2021)

- Major improvements in [Form compiler parameters](chapter4/compiler_parameters), using pandas and seaborn for visualization of speed-ups gained using form compiler parameters.
- API change: `dolfinx.cpp.la.scatter_forward(u.x)` -> `u.x.scatter_forward`
- Various plotting updates due to new version of pyvista.
- Updating of the [Hyperelasticity demo](chapter2/hyperelasticity), now using DOLFINx wrappers to create the non-linear problem
- Internal updates due to bumping of jupyter-book versions
- Various typos and capitalizations fixed by @mscroggs in [PR 35](https://github.com/jorgensd/dolfinx-tutorial/pull/35).

## 0.1.0 (11.05.2021)

- First tagged release of DOLFINx Tutorial, compatible with [DOLFINx 0.1.0](https://github.com/FEniCS/dolfinx/releases/tag/0.1.0).
