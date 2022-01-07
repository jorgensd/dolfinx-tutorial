FROM dolfinx/lab

WORKDIR /tmp/

# Install h5py
RUN  HDF5_MPI="ON" HDF5_DIR="/usr/lib/x86_64-linux-gnu/hdf5/mpich/" CC=mpicc pip3 install --no-cache-dir --no-binary=h5py h5py

# Install jupyter dependencies
RUN pip3 install setuptools jupyterhub jupyterlab nbconvert --upgrade

# Install meshio
RUN pip3 install --no-cache-dir  --upgrade meshio
# Install pandas and seaborn
RUN pip3 install --no-cache-dir --upgrade setuptools seaborn pandas

# Install pyvista
RUN pip3 install --upgrade  matplotlib pyvista

RUN pip3 install jupyterlab --upgrade

# Install plotting dependencies
RUN pip3 install pythreejs ipygany
RUN jupyter nbextension enable --py --sys-prefix ipygany

# Install xvfb for screenshots
RUN apt-get -qq update && \
    apt-get install -y libgl1-mesa-dev xvfb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install progress-bar
RUN pip3 install tqdm pygments --upgrade


ENTRYPOINT []