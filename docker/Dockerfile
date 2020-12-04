# Dockerfile describing development builds of FEniCS-X
#
# Authors: Jack S. Hale <jack.hale@uni.lu> Lizao Li
# <lzlarryli@gmail.com> Garth N. Wells <gnw20@cam.ac.uk> Jan Blechta
# <blechta@karlin.mff.cuni.cz>


ARG GMSH_VERSION=4.6.0
ARG PYBIND11_VERSION=2.6.0
ARG PETSC_VERSION=3.14.1
ARG SLEPC_VERSION=3.14.0
ARG PETSC4PY_VERSION=3.14.0
ARG SLEPC4PY_VERSION=3.14.0
# Should be updated upon a new KaHIP release
ARG KAHIP_VERSION=14be06c

########################################

FROM ubuntu:20.04 as dev-env
LABEL maintainer="Jørgen  S. Dokken <jsd55@cam.ac.uk>"
LABEL description="FEniCS-X tutorials image (dolfinx complex 32bit mode)"

ARG GMSH_VERSION
ARG PYBIND11_VERSION
ARG PETSC_VERSION
ARG PETSC4PY_VERSION
ARG SLEPC_VERSION
ARG SLEPC4PY_VERSION

# The following ARGS are used in the dev-env layer.
# They are safe defaults. They can be overridden by the user.
# Compiler optimisation flags for SLEPc and PETSc, all languages.
ARG PETSC_SLEPC_OPTFLAGS="-O2"
# Turn on PETSc and SLEPc debugging. "yes" or "no".
ARG PETSC_SLEPC_DEBUGGING="no"
# Ubuntu MPI variant. "mpich" or "openmpi".
ARG MPI="mpich"

ARG NB_USER=fenics
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

WORKDIR /tmp


# Environment variables
ENV OPENBLAS_NUM_THREADS=1 \
    OPENBLAS_VERBOSE=0

# Install dependencies available via apt-get.
# - First set of packages are required to build and run FEniCS.
# - Second set of packages are recommended and/or required to build
#   documentation or tests.
# - Third set of packages are optional, but required to run gmsh
#   pre-built binaries.
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get -qq update && \
    apt-get -yq --with-new-pkgs -o Dpkg::Options::="--force-confold" upgrade && \
    apt-get -y install \
    clang-10 \
    cmake \
    g++ \
    gfortran \
    libboost-dev \
    libboost-filesystem-dev \
    libboost-timer-dev \
    libeigen3-dev \
    libhdf5-${MPI}-dev \
    liblapack-dev \
    lib${MPI}-dev \
    libopenblas-dev \
    ninja-build \
    pkg-config \
    python3-dev \
    python3-matplotlib \
    python3-numpy \
    python3-pip \
    python3-scipy \
    python3-setuptools && \
    #
    apt-get -y install \
    doxygen \
    git \
    graphviz \
    valgrind \
    wget && \
    #
    apt-get -y install \
    libglu1 \
    libxcursor-dev \
    libxinerama1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


# Install Python packages (via pip)
# - First set of packages are required to build and run DOLFINX Python.
# - Second set of packages are recommended and/or required to build
#   documentation or run tests.
RUN pip3 install --no-cache-dir mpi4py numba && \
    pip3 install --no-cache-dir cffi cppimport flake8 pytest pytest-xdist sphinx sphinx_rtd_theme

# Install pybind11
RUN wget -nc --quiet https://github.com/pybind/pybind11/archive/v${PYBIND11_VERSION}.tar.gz && \
    tar -xf v${PYBIND11_VERSION}.tar.gz && \
    cd pybind11-${PYBIND11_VERSION} && \
    mkdir build && \
    cd build && \
    cmake -DPYBIND11_TEST=False ../ && \
    make install && \
    rm -rf /tmp/*

# Install KaHIP
RUN cd /usr/local && \
    git clone https://github.com/schulzchristian/KaHIP.git && \
    cd KaHIP/ && \
    git checkout ${KAHIP_VERSION} && \
    ./compile_withcmake.sh

ENV KAHIP_ROOT=/usr/local/KaHIP

# Download Install Gmsh SDK
RUN cd /usr/local && \
    wget -nc --quiet http://gmsh.info/bin/Linux/gmsh-${GMSH_VERSION}-Linux64-sdk.tgz && \
    tar -xf gmsh-${GMSH_VERSION}-Linux64-sdk.tgz && \
    rm gmsh-${GMSH_VERSION}-Linux64-sdk.tgz

ENV PATH=/usr/local/gmsh-${GMSH_VERSION}-Linux64-sdk/bin:$PATH \
    PYTHONPATH=/usr/local/gmsh-${GMSH_VERSION}-Linux64-sdk/lib:$PYTHONPATH

# Install PETSc with real and complex types
ENV PETSC_DIR=/usr/local/petsc SLEPC_DIR=/usr/local/slepc
WORKDIR /tmp
RUN apt-get -qq update && \
    apt-get -y install bison flex && \
    wget -nc --quiet http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-lite-${PETSC_VERSION}.tar.gz -O petsc-${PETSC_VERSION}.tar.gz && \
    mkdir -p ${PETSC_DIR} && tar -xf petsc-${PETSC_VERSION}.tar.gz -C ${PETSC_DIR} --strip-components 1 && \
    cd ${PETSC_DIR} && \
    # Complex, 32-bit int
    python3 ./configure \
    PETSC_ARCH=linux-gnu-complex-32 \
    --COPTFLAGS="${PETSC_SLEPC_OPTFLAGS}" \
    --CXXOPTFLAGS="${PETSC_SLEPC_OPTFLAGS}" \
    --FOPTFLAGS="${PETSC_SLEPC_OPTFLAGS}" \
    --with-64-bit-indices=no \
    --with-debugging=${PETSC_SLEPC_DEBUGGING} \
    --with-fortran-bindings=no \
    --with-shared-libraries \
    --download-blacs \
    --download-metis \
    --download-mumps \
    --download-ptscotch \
    --download-scalapack \
    --download-suitesparse \
    --download-superlu \
    --download-superlu_dist \
    --with-scalar-type=complex && \
    make PETSC_DIR=/usr/local/petsc PETSC_ARCH=linux-gnu-complex-32 ${MAKEFLAGS} all

# Install SLEPc
WORKDIR /tmp

RUN wget -nc --quiet https://slepc.upv.es/download/distrib/slepc-${SLEPC_VERSION}.tar.gz -O slepc-${SLEPC_VERSION}.tar.gz && \
    mkdir -p ${SLEPC_DIR} && tar -xf slepc-${SLEPC_VERSION}.tar.gz -C ${SLEPC_DIR} --strip-components 1 && \
    cd ${SLEPC_DIR} && \
    export PETSC_ARCH=linux-gnu-complex-32 && \
    python3 ./configure && \
    make && \
    rm -rf ${SLEPC_DIR}/CTAGS ${SLEPC_DIR}/TAGS ${SLEPC_DIR}/docs ${SLEPC_DIR}/src/ ${SLEPC_DIR}/**/obj/ ${SLEPC_DIR}/**/test/ && \
    rm -rf /tmp/*

# Install petsc4py and slepc4py with real and complex types
RUN PETSC_ARCH=linux-gnu-complex-32 pip3 install --no-cache-dir petsc4py==${PETSC4PY_VERSION} slepc4py==${SLEPC4PY_VERSION}

WORKDIR ${HOME}


RUN git clone https://github.com/FEniCS/dolfinx.git && \
    git clone https://github.com/FEniCS/fiat.git && \
    git clone https://github.com/FEniCS/ufl.git && \
    git clone https://github.com/FEniCS/ffcx.git && \
    cd fiat &&\
    pip3 install --no-cache-dir . &&\
    cd ..  &&\
    cd ufl &&\
    pip3 install --no-cache-dir . &&\
    cd .. &&\
    cd ffcx &&\
    pip3 install --no-cache-dir . &&\
    cd ..
RUN cd dolfinx &&\
    mkdir build && \
    cd build && \
    PETSC_ARCH=linux-gnu-complex-32 cmake -G Ninja -DCMAKE_INSTALL_PREFIX=/usr/local/dolfinx-complex -DCMAKE_BUILD_TYPE=${DOLFINX_CMAKE_BUILD_TYPE} -DCMAKE_CXX_FLAGS=${DOLFIN_CMAKE_CXX_FLAGS} ../cpp && \
    ninja install && \
    . /usr/local/dolfinx-complex/share/dolfinx/dolfinx.conf && \
    cd ../python && \
    CXXFLAGS=${DOLFINX_CMAKE_CXX_FLAGS} PETSC_ARCH=linux-gnu-complex-32 pip3 install --target /usr/local/dolfinx-complex/lib/python3.8/dist-packages --no-dependencies . && \
    cd ../ && \
    rm -r build &&\
    cd ..


# The following ARGS are used in the dolfinx layer.
# They are safe defaults.
# CMake build type for DOLFINX C++ build. See CMake documentation.
ARG DOLFINX_CMAKE_BUILD_TYPE="RelWithDebInfo"
# Extra CMake C++ compiler flags for DOLFINX C++ build.
ARG DOLFINX_CMAKE_CXX_FLAGS


# Note that because we inherit from dev-env we do not inherit these ENV from
# dolfinx-onbuild so this must be repeated here.
ENV PKG_CONFIG_PATH=/usr/local/dolfinx-complex/lib/pkgconfig:$PKG_CONFIG_PATH \
    PETSC_ARCH=linux-gnu-complex-32 \
    PYTHONPATH=/usr/local/dolfinx-complex/lib/python3.8/dist-packages:$PYTHONPATH \
    LD_LIBRARY_PATH=/usr/local/dolfinx-complex/lib:$LD_LIBRARY_PATH