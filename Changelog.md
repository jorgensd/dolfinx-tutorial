# Changelog

## Dev
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