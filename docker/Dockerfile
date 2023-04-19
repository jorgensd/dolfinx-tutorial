FROM ghcr.io/fenics/dolfinx/lab:nightly

WORKDIR /tmp/

ENV DEB_PYTHON_INSTALL_LAYOUT=deb_system
ENV HDF5_MPI="ON"
ENV HDF5_DIR="/usr/local"
ENV PYVISTA_JUPYTER_BACKEND="panel"

# Requirements for pyvista
RUN apt-get update && apt-get install -y libgl1-mesa-glx libxrender1 xvfb nodejs

# Upgrade setuptools and pip
RUN python3 -m pip install setuptools pip pkgconfig

# Install `h5py`
# https://github.com/h5py/h5py/issues/2222
RUN python3 -m pip install cython
RUN python3 -m pip install --no-build-isolation --no-binary=h5py h5py

ADD requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r requirements.txt
RUN python3 -m pip cache purge
RUN jupyter lab build
ENTRYPOINT ["jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
