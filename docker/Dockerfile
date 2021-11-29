FROM dolfinx/lab

WORKDIR /tmp/

# Install meshio
RUN  HDF5_MPI="ON" HDF5_DIR="/usr/lib/x86_64-linux-gnu/hdf5/mpich/" CC=mpicc pip3 install --no-cache-dir --no-binary=h5py h5py meshio

# Dependencies for pyvista and related packages
RUN wget -qO - https://deb.nodesource.com/setup_17.x | bash && \
    apt-get -qq update && \
    apt-get install -y libgl1-mesa-dev xvfb nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip3 install setuptools jupyterhub nbconvert --upgrade

# Pyvista ITKWidgets dependencies
RUN pip3 install --no-cache-dir matplotlib itkwidgets ipywidgets ipyvtklink pyvista seaborn pandas
RUN jupyter labextension install jupyter-matplotlib jupyterlab-datawidgets itkwidgets && \
    rm -rf /usr/local/share/.cache/*

# Install progress-bar
RUN pip3 install --no-cache-dir tqdm pygments --upgrade

ENTRYPOINT []