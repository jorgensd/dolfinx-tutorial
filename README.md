# The DOLFINx tutorial
![CI status](https://github.com/jorgensd/dolfinx-tutorial/actions/workflows/build-publish.yml/badge.svg)
![Main status](https://github.com/jorgensd/dolfinx-tutorial/actions/workflows/main-test.yml/badge.svg)

Author: Jørgen S. Dokken

This is the source code for the dolfinx-tutorial [webpage](https://jorgensd.github.io/dolfinx-tutorial/).
If you have any comments, corrections or questions, please submit an issue in the issue tracker.

## Contributing
If you want to contribute to this tutorial, please make a fork of the repository, make your changes, and test that the CI passes. You can do this locally by downloading [act](https://github.com/nektos/act) and call
```bash
act -j test-against-master
```
Any code added to the tutorial should work in parallel.

Alternatively, if you want to add a separate chapter, a Jupyter notebook can be added to a pull request, without integrating it into the tutorial. If so, the notebook will be reviewed and modified to be included in the tutorial.

# Docker images
Docker images for this tutorial can be found at [Docker hub](https://hub.docker.com/repository/docker/dokken92/dolfinx_custom)

Additional requirements on top of the `dolfinx/lab` images can be found at [Dockerfile](docker/Dockerfile) and [requirements.txt](docker/requirements.txt)

##
An image building DOLFINx, Basix and FFCx from source can be built using:
```bash
docker built -f docker/LocalDockerFile -t local_lab_env .
```
and run
```bash
docker run --rm -v $PWD:/root/shared -w /root/shared  --init -p 8888:8888 local_lab_env
```
