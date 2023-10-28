# The DOLFINx tutorial
[![Test, build and publish](https://github.com/jorgensd/dolfinx-tutorial/actions/workflows/build-publish.yml/badge.svg)](https://github.com/jorgensd/dolfinx-tutorial/actions/workflows/build-publish.yml)
[![Test release branch against DOLFINx nightly build](https://github.com/jorgensd/dolfinx-tutorial/actions/workflows/nightly.yml/badge.svg)](https://github.com/jorgensd/dolfinx-tutorial/actions/workflows/nightly.yml)

Author: Jørgen S. Dokken

This is the source code for the dolfinx-tutorial [webpage](https://jorgensd.github.io/dolfinx-tutorial/).
If you have any comments, corrections or questions, please submit an issue in the issue tracker.

## Contributing
If you want to contribute to this tutorial, please make a fork of the repository, make your changes, and test that the CI passes. You can do this locally by downloading [act](https://github.com/nektos/act) and call
```bash
act -j test-nightly
```
Any code added to the tutorial should work in parallel.

Alternatively, if you want to add a separate chapter, a Jupyter notebook can be added to a pull request, without integrating it into the tutorial. If so, the notebook will be reviewed and modified to be included in the tutorial.

Note that every chapter is written as an IPython notebook and a Python file. These are kept in sync by jupytext. See [their notes](https://jupytext.readthedocs.io/en/latest/install.html#jupytext-commands-in-jupyterlab) on how to keep them in sync.

# Docker images
Docker images for this tutorial can be found in the [packages tab](https://github.com/jorgensd/dolfinx-tutorial/pkgs/container/dolfinx-tutorial) 

Additional requirements on top of the `dolfinx/lab:nightly` images can be found at [Dockerfile](docker/Dockerfile) and [requirements.txt](docker/requirements.txt)
