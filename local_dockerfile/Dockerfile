FROM dolfinx/lab

# Install meshio
RUN HDF5_MPI="ON" CC=mpicc HDF5_DIR="/usr/lib/x86_64-linux-gnu/hdf5/mpich/" pip3 install --no-cache-dir --no-binary=h5py h5py meshio

# Pyvista ITKWidgets dependencies
RUN pip3 install --no-cache-dir --upgrade setuptools itkwidgets ipywidgets matplotlib pyvista ipyvtklink seaborn pandas

# Dependencies for pyvista and related packages
RUN wget -qO - https://deb.nodesource.com/setup_17.x | bash && \
    apt-get -qq update && \
    apt-get install -y libgl1-mesa-dev xvfb nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN jupyter labextension install jupyter-matplotlib jupyterlab-datawidgets itkwidgets


# Install progress-bar
RUN pip3 install tqdm pygments --upgrade

EXPOSE 8888/tcp
ENV SHELL /bin/bash

ENTRYPOINT ["jupyter", "notebook", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]