# The dolfin-X tutorial
![CI status](https://github.com/jorgensd/dolfinx-tutorial/actions/workflows/blank.yml/badge.svg)
![Main status](https://github.com/jorgensd/dolfinx-tutorial/actions/workflows/main-test.yml/badge.svg)

This is the github repo for the dolfinx-tutorial [webpage](https://jorgensd.github.io/dolfinx-tutorial/).
If you have any comments, corrections or questions, please submit an issue in the issue tracker.

## Contributing
If you want to contribute to this tutorial, please make a fork of the repository, make your changes, and test that it builds correctly using the build command locally in your computer:
```bash
PYVISTA_OFF_SCREEN=false jupyter-book build  -W .
```
Any code added to the tutorial should work in parallel.

Alternatively, if you want to add a separate chapter, a jupyter notebook can be added to a pull request, without integrating it into the tutorial. If so, the notebook will be reviewed and modified to be included in the tutorial.

## Requirements for dockerfile
To create a suitable docker-file for the Binder containers to build from, see the instructions at.
https://mybinder.readthedocs.io/en/latest/tutorials/dockerfile.html


Push book (deprecated as github actions does this for you)
```bash
pip3 install ghp-import
ghp-import -n -p -f _build/html
```
