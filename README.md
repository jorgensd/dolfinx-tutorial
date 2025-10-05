# The DOLFINx tutorial

[![Test, build and publish](https://github.com/jorgensd/dolfinx-tutorial/actions/workflows/deploy.yml/badge.svg)](https://github.com/jorgensd/dolfinx-tutorial/actions/workflows/deploy.yml)
[![Test release branch against DOLFINx nightly build](https://github.com/jorgensd/dolfinx-tutorial/actions/workflows/test_nightly.yml/badge.svg)](https://github.com/jorgensd/dolfinx-tutorial/actions/workflows/test_nightly.yml)

Author: Jørgen S. Dokken

This is the source code for the dolfinx-tutorial [webpage](https://jorgensd.github.io/dolfinx-tutorial/).
If you have any comments, corrections or questions, please submit an issue in the issue tracker.

## Contributing

If you want to contribute to this tutorial, please make a fork of the repository, make your changes, and test that the CI passes. 

Alternatively, if you want to add a separate chapter, a Jupyter notebook can be added to a pull request, without integrating it into the tutorial. If so, the notebook will be reviewed and modified to be included in the tutorial.

Any code added to the tutorial should work in parallel. If any changes are made to `ipynb` files, please ensure that these changes are reflected in the corresponding `py` files by using [`jupytext`](https://jupytext.readthedocs.io/en/latest/faq.html#can-i-use-jupytext-with-jupyterhub-binder-nteract-colab-saturn-or-azure):


## Building the book and running code
The book is built using [jupyterbook](https://jupyterbook.org/). The following environment variables should be set if you want to build the book
```bash
PYVISTA_OFF_SCREEN=false
PYVISTA_JUPYTER_BACKEND="html"
JUPYTER_EXTENSION_ENABLED=true
LIBGL_ALWAYS_SOFTWARE=1
```

If you run the tutorial using `jupyter-lab`, for instance through `conda`, one should set the following environment variables
```bash
PYVISTA_OFF_SCREEN=false
PYVISTA_JUPYTER_BACKEND="trame"
JUPYTER_EXTENSION_ENABLED=true
LIBGL_ALWAYS_SOFTWARE=1
```
If you use docker to run your code, you should set the following variables:
```bash
docker run -ti -e DISPLAY=$DISPLAY -e LIBGL_ALWAYS_SOFTWARE=1 -e PYVISTA_OFF_SCREEN=false -e PYVISTA_JUPYTER_BACKEND="trame" -e JUPYTER_EXTENSION_ENABLED=true --network=host -v $(pwd):/root/shared -w /root/shared  ....
```

To run python scripts, either choose `PYVISTA_OFF_SCREEN=True` to get screenshots, or render interactive plots with `PYVISTA_OFF_SCREEN=False`


```bash
python3 -m jupytext --sync  */*.ipynb --set-formats ipynb,py:light
```
or
```bash
python3 -m jupytext --sync  */*.py --set-formats ipynb,py:light
```

Any code added to the tutorial should work in parallel.

To strip notebook output, one can use pre-commit.

```bash
pre-commit run --all-files
```

## Dependencies

It is adviced to use a pre-installed version of DOLFINx, for instance through conda or docker. Remaining dependencies can be installed with

```bash
python3 -m pip install --no-binary=h5py -e .
```

# Docker images

Docker images for this tutorial can be found in the [packages tab](https://github.com/jorgensd/dolfinx-tutorial/pkgs/container/dolfinx-tutorial)

Additional requirements on top of the `dolfinx/lab:nightly` images can be found at [Dockerfile](docker/Dockerfile) and [pyproject.toml](./pyproject.toml)

##

An image building DOLFINx, Basix, UFL and FFCx from source can be built using:

```bash
docker build -f ./docker/Dockerfile -t local_lab_env .
```

from the root of this repository, and run

```bash
 docker run --rm -ti -v $(pwd):/root/shared -w /root/shared  --init -p 8888:8888 local_lab_env
```

from the main directory.
