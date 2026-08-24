# Execute from root of repo as: docker buildx build --platform=linux/arm64,linux/amd64 -f docker/Dockerfile . --progress=plain

FROM ghcr.io/fenics/dolfinx/lab:stable

ARG TARGETPLATFORM


ENV DEB_PYTHON_INSTALL_LAYOUT=deb_system
ENV HDF5_MPI="ON"

WORKDIR /tmp/
# Requirements for pyvista (gl1 and render1) and jupyterlab (nodejs and curl)
RUN apt-get update && apt-get install -y libgl1-mesa-dev mesa-utils curl


# Install netgen from source
RUN  apt-get update && \
    apt-get -y install python3 python3-tk libpython3-dev libxmu-dev tk-dev tcl-dev cmake git g++ libglu1-mesa-dev liblapacke-dev libocct-data-exchange-dev libocct-draw-dev occt-misc libtbb-dev libxi-dev
WORKDIR /ngsuite
RUN git clone --branch=v6.2.2505 --single-branch https://github.com/NGSolve/netgen.git netgen-src
WORKDIR /ngsuite/netgen-src
RUN git submodule update --init --recursive
WORKDIR /ngsuite
RUN mkdir netgen-build
RUN mkdir netgen-install
RUN cmake -DCMAKE="-cxx-flags=-flax-vector-conversions" -DCMAKE_INSTALL_PREFIX=/ngsuite/netgen-install /ngsuite/netgen-src
RUN make
RUN make install
ENV NETGENDIR=/ngsuite/netgen-install/bin
ENV PATH=$NETGENDIR:$PATH
ENV PYTHONPATH=$NETGENDIR/../lib/python3.12/site-packages:${PYTHONPATH}:${PYTHONPATH}:$PATH

# Install vtk
RUN python3 -m pip install vtk

RUN ARCH_TRIPLET=$(gcc -print-multiarch) && \
    export CC="mpicc" && \
    export HDF5_MPI="ON" && \
    export HDF5_INCLUDEDIR="/usr/include/hdf5/openmpi" && \
    export HDF5_LIBDIR="/usr/lib/${ARCH_TRIPLET}/hdf5/openmpi" && \
    pip install --no-binary=h5py h5py --no-build-isolation --no-cache-dir

ADD pyproject.toml /tmp/pyproject.toml
WORKDIR /tmp
# As we install netgen from source we don't need the netgen deps here
RUN python3 -m pip install --no-cache-dir --no-binary=h5py --no-build-isolation -v .
RUN python3 -m pip cache purge
RUN python3 -m pip install poetry_dynamic_versioning poetry-core
RUN python3 -m pip install ngsPETSc --no-build-isolation --no-deps

ENV PYVISTA_OFF_SCREEN="false"
ENV PYVISTA_JUPYTER_BACKEND="trame"
ENV LIBGL_ALWAYS_SOFTWARE=1


ENTRYPOINT ["jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
