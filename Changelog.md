# Changelog

## Dev 
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