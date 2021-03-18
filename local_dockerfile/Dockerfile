FROM dolfinx/lab

# Dependencies for pyvista and related packages
RUN wget -qO - https://deb.nodesource.com/setup_15.x | bash && \
    apt-get -qq update && \
    apt-get install -y libgl1-mesa-dev xvfb nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Pyvista ITKWidgets dependencies
RUN pip3 install --no-cache-dir itkwidgets ipywidgets matplotlib
RUN jupyter labextension install jupyter-matplotlib jupyterlab-datawidgets itkwidgets

# Install meshio
RUN pip3 install --no-cache-dir --no-binary=h5py h5py meshio

# Install progress-bar
RUN pip3 install tqdm

# TMP install gmsh
ARG GMSH_VERSION=4.8.0

RUN cd /usr/local && \
    wget -nc --quiet http://gmsh.info/bin/Linux/gmsh-${GMSH_VERSION}-Linux64-sdk.tgz && \
    tar -xf gmsh-${GMSH_VERSION}-Linux64-sdk.tgz && \
    rm gmsh-${GMSH_VERSION}-Linux64-sdk.tgz && \
    rm -r gmsh-4.6.0-Linux64-sdk

ENV PATH=/usr/local/gmsh-${GMSH_VERSION}-Linux64-sdk/bin:$PATH \
    PYTHONPATH=/usr/local/gmsh-${GMSH_VERSION}-Linux64-sdk/lib:$PYTHONPATH

EXPOSE 8888/tcp
ENV SHELL /bin/bash

ENTRYPOINT ["jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]