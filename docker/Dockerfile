FROM dolfinx/lab

WORKDIR /tmp/

# Dependencies for pyvista and related packages
RUN wget -qO - https://deb.nodesource.com/setup_15.x | bash && \
    apt-get -qq update && \
    apt-get install -y libgl1-mesa-dev xvfb nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Pyvista ITKWidgets dependencies
RUN pip3 install --no-cache-dir matplotlib itkwidgets ipywidgets
RUN jupyter labextension install jupyter-matplotlib jupyterlab-datawidgets itkwidgets && \
    rm -rf /usr/local/share/.cache/*

# Temporary install gmsh
ARG GMSH_VERSION=4.8.0

RUN cd /usr/local && \
    wget -nc --quiet http://gmsh.info/bin/Linux/gmsh-${GMSH_VERSION}-Linux64-sdk.tgz && \
    tar -xf gmsh-${GMSH_VERSION}-Linux64-sdk.tgz && \
    rm gmsh-${GMSH_VERSION}-Linux64-sdk.tgz

ENV PATH=/usr/local/gmsh-${GMSH_VERSION}-Linux64-sdk/bin:$PATH \
    PYTHONPATH=/usr/local/gmsh-${GMSH_VERSION}-Linux64-sdk/lib:$PYTHONPATH

# Install meshio
RUN pip3 install --no-cache-dir --no-binary=h5py h5py meshio

# Install progress-bar
RUN pip3 install --no-cache-dir tqdm

ENTRYPOINT []