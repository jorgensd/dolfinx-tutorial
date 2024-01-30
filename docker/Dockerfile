# Execute from root of repo as: docker buildx build --platform=linux/arm64,linux/amd64 -f docker/Dockerfile ./docker/ --progress=plain

FROM ghcr.io/fenics/dolfinx/lab:v0.7.3
ARG TARGETPLATFORM


ENV DEB_PYTHON_INSTALL_LAYOUT=deb_system
ENV HDF5_MPI="ON"
ENV HDF5_DIR="/usr/local"
ENV PYVISTA_JUPYTER_BACKEND="panel"
WORKDIR /tmp/
# Requirements for pyvista (gl1 and render1) and jupyterlab (nodejs and curl)
RUN apt-get update && apt-get install -y libgl1-mesa-glx libxrender1 xvfb curl
RUN curl -sL https://deb.nodesource.com/setup_18.x -o nodesource_setup.sh && \
    bash nodesource_setup.sh && \
    apt install nodejs

# Upgrade setuptools and pip
RUN python3 -m pip install -U setuptools pip pkgconfig

# BUILD VTK
# ENV VTK_VERSION="v9.2.6"
# RUN git clone --recursive --branch ${VTK_VERSION} --single-branch https://gitlab.kitware.com/vtk/vtk.git vtk && \
#     cmake -G Ninja -DVTK_WHEEL_BUILD=ON -DVTK_WRAP_PYTHON=ON vtk/ && \
#     ninja && \
#     python3 setup.py bdist_wheela

RUN echo ${TARGETPLATFORM}
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then python3 -m pip install "https://github.com/finsberg/vtk-aarch64/releases/download/vtk-9.2.6-cp310/vtk-9.2.6.dev0-cp310-cp310-linux_aarch64.whl"; fi
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then python3 -m pip install vtk; fi

ADD pyproject.toml /tmp/pyproject.toml
RUN python3 -m pip install --no-cache-dir --no-binary=h5py -v .
RUN python3 -m pip cache purge
RUN jupyter lab build
ENTRYPOINT ["jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
