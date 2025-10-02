# Execute from root of repo as: docker buildx build --platform=linux/arm64,linux/amd64 -f docker/Dockerfile ./docker/ --progress=plain

FROM ghcr.io/fenics/dolfinx/lab:nightly
ARG TARGETPLATFORM


ENV DEB_PYTHON_INSTALL_LAYOUT=deb_system
ENV HDF5_MPI="ON"
ENV HDF5_DIR="/usr/local"
ENV PYVISTA_JUPYTER_BACKEND="static"

WORKDIR /tmp/
# Requirements for pyvista (gl1 and render1) and jupyterlab (nodejs and curl)
RUN apt-get update && apt-get install -y libgl1-mesa-dev libxrender1 xvfb curl
RUN curl -sL https://deb.nodesource.com/setup_18.x -o nodesource_setup.sh && \
    bash nodesource_setup.sh && \
    apt -y install nodejs

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

ADD pyproject.toml /tmp/pyproject.toml
WORKDIR /tmp
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then python3 -m pip install --no-cache-dir --no-binary=h5py -v .; fi
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then python3 -m pip install --no-cache-dir --no-binary=h5py -v .[netgen]; fi

RUN python3 -m pip install -v ngsPETSc@git+https://github.com/NGSolve/ngsPETSc.git scipy --no-deps
RUN python3 -m pip cache purge

ENTRYPOINT ["jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
