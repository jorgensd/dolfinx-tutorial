# The FEniCS-X tutorial
This webpage gives a concise overview of the functionality of [Dolfin-X](https://github.com/FEniCS/dolfinx/), including a gentle introduction to the finite element method. This webpage is an adaptation of the [FEniCS tutorial](https://www.springer.com/gp/book/9783319524610).

Dolfin-X can be used as either a C++ or Python software, but this tutorial will focus on Python programming, as it is the simplest and most effective approach for beginners. After having gone through this tutorial, the reader should familiarize themselfs with the Dolfin-X [documentation](https://fenicsproject.org/docs/dolfinx/dev/python/), which includes the API and numerous demos.

Comments and corrections to this webpage should be submitted to [dolfinx-tutorial issue tracker](https://github.com/jorgensd/dolfinx-tutorial/issues).

## The FEniCS Project

The FEniCS project is a research and software project aimed at creating mathematical methods and software for solving partial differential equations. This includes creating intuitive, efficient and flexible software. The project was initiated in 2003, and is developed in collaboration between researchers from a number of universities and research institutes around the world. For the latest updates and more information about the FEniCS project, visit the [FEniCS](https://fenicsproject.org) webpage.

The latest version of the FEniCS project, FEniCS-X, consists of several building blocks, namely [FIAT](https://github.com/FEniCS/fiat)/[LibTab](https://github.com/FEniCS/libtab), [UFL](https://github.com/FEniCS/ufl), [FFC-X](https://github.com/FEniCS/ffcx) and [Dolfin-X](https://github.com/FEniCS/dolfinx). We will now go through the main objectives of each of these building blocks. Dolfin-X is the high performance C++ backend of FEniCS-X, where structures such as meshes, function spaces and functions are implemented. 
Additionally, Dolfin-X also contains compute intensive functions such as finite element assembly and mesh refinement algorithms. It also provides an interface to linear algebra solvers and data-structures, such as [PETSc](https://www.mcs.anl.gov/petsc/). UFL is a high-level form language for describing variational formulations with a high-level mathematical syntax. FFC-X is the form compiler of FEniCS; given variational formulations written with UFL, it generates efficient C code. FIAT/LibTab is the finite element backend of FEniCS, responsible for generating finite element basis functions. 

## What you will learn

The goal of this tutorial is to demonstrate how to apply the finite element to solve PDEs in FEniCS. Through a series of examples, we will demonstrate how to:

- Solve linear PDEs (such as the Poisson equation),
- Solve time-dependent PDEs (such as the heat equation),
- Solve nonlinear PDEs,
- Solve systems of time-dependent nonlinear PDEs.

Important topics involve how to set boundary conditions of various types (Dirichlet, Neumann, Robin), how to create meshes, how to define variable coefficients, how to interact with linear and nonlinear solvers, and how to postprocess and visualize solutions.

# Working with this tutorial

Most of the mathematical part of the examples will be kept at a simple level, such that we can keep the focus on the functionality and syntax of FEniCS-X. Therefore we will mostly use the Poisson equation and the time-dependent diffusion equation as model problems. We will use adjusted input data, such that the solution of the problem can be exactly reproduced on uniform, structured meshes with the finite element method. This greatly simplifies the verification of the implementations. 
Occasionally we will consider a more physically relevant example to remind the reader that there are no big leaps from solving simple model problems to challenging real-world problems when using FEniCS-X.

## Obtaining the software

Working with this tutorial obviously require access to the FEniCS software. The software is quite complex, and building the software and all the dependencies from source can be a daunting task. The list of dependencies can be found at [https://fenicsproject.org/docs/dolfinx/dev/python/installation.html](https://fenicsproject.org/docs/dolfinx/dev/python/installation.html).


## Docker
Fortunately, we supply a pre-built docker image at [https://hub.docker.com/r/dolfinx/dolfinx](https://hub.docker.com/r/dolfinx/dolfinx).
The [Dockerfile](https://github.com/FEniCS/dolfinx/blob/master/Dockerfile)
provides a definitive build recipe. 

Docker is a software that uses \textit{containers} to supply software across different kinds of operating systems (Linux, Mac, Windows). The first step is to install docker, following the instructions at their [web-page](https://docs.docker.com/get-started/). 
As the dolfinx docker images are hosted at Docker-hub, one can directly access the image 
```
docker run dolfinx/dolfin
```
There are several ways of customizing a docker container, such as mounting volumes/sharing folder, setting a working directory, sharing graphical interfaces etc. See `docker run --help` for an extensive list.

Once you have installed dolfin-X, either by using docker or installing form source, you can test the installation by running `python3 -c "import dolfinx"`. If all goes well, no error-messages should appear.

If you are a beginner in Python, we suggest reading [A Primer on Scientific Programming in Python](https://link.springer.com/book/10.1007%2F978-3-662-49887-3) by Hans Petter Langtangen, which will give you a gentle introduction to the Python programming language. Note that dolfin-X, being a state of the art finite element solver, only supports Python 3, as Python 2 reached its end of life January 1st, 2020. To automatically transfer Python 2 scripts to Python 3, it is suggested to use the [2to3](https://docs.python.org/3/library/2to3.html)-package, which provides automated translation of the code.


## The Finite element method
There has been written many good text-books on the finite element method, and we refer to the original FEniCS tutorial, for references to these, (see Chapter 1.6.2 of [The FEniCS tutorial](https://www.springer.com/gp/book/9783319524610)).


