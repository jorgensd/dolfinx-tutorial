FROM dolfinx/lab:v0.6.0

WORKDIR /tmp/

# Requirements for pyvista/trame
RUN apt-get update && apt-get install -y libgl1-mesa-glx libxrender1 xvfb

# Upgrade setuptools and pip
# https://github.com/pypa/setuptools/issues/3269#issuecomment-1254507377
ENV DEB_PYTHON_INSTALL_LAYOUT=deb_system
RUN python3 -m pip install -U setuptools pip pkgconfig

# Install `h5py`
# https://github.com/hl5py/h5py/issues/2222
RUN python3 -m pip install cython
RUN python3 -m pip install --no-build-isolation --no-binary=h5py h5py

ADD requirements.txt /tmp/requirements.txt

ENV PYVISTA_TRAME_SERVER_PROXY_PREFIX='/proxy/'
ENV PYVISTA_TRAME_SERVER_PROXY_ENABLED="True"


# Install dependencies
RUN HDF5_MPI="ON" HDF5_DIR="/usr/local/" CC=mpicc pip3 install --no-cache-dir -r requirements.txt --upgrade && pip3 cache purge

ENTRYPOINT ["jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
