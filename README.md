# The dolfin-X tutorial

This is the github repo for the dolfinx-tutorial [webpage](https://jorgensd.github.io/dolfinx-tutorial/).
If you have any comments, corrections or questions, please submit an issue in the issue tracker.

## Contributing
If you want to contribute to this tutorial, please make a fork of the repository, make your changes, and test that it builds correctly using the build command locally in your computer:
```bash
pip3 install -U jupyter-book phinxcontrib-bibtex==1.0.0
jupyter-book build .
```
Any code added to the tutorial should work in parallel.

## Requirements for dockerfile
To create a suitable docker-file for the Binder containers to build from, see the instructions at.
https://mybinder.readthedocs.io/en/latest/tutorials/dockerfile.html


Push book (deprecated as github actions does this for you)
```bash
pip3 install ghp-import
ghp-import -n -p -f _build/html
```
