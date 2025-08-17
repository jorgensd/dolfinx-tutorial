# Execute from root of repo as: docker buildx build --platform=linux/arm64,linux/amd64 -f docker/Dockerfile ./docker/ --progress=plain

FROM ghcr.io/fenics/dolfinx/lab:v0.9.0
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

# Upgrade setuptools and pip
RUN python3 -m pip install -U setuptools pip pkgconfig poetry-core


RUN python3 -m pip install vtk

ADD pyproject.toml /tmp/pyproject.toml
RUN python3 -m pip install --no-cache-dir --no-binary=h5py -v .
RUN python3 -m pip cache purge

ENTRYPOINT ["jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
