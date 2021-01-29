FROM dolfinx/dolfinx

# Install visualization tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1-mesa-dev xvfb && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install meshio
RUN pip3 install --no-cache-dir --no-binary=h5py h5py meshio

# Install notebook package
RUN pip3 install --no-cache --upgrade pip && \
    pip3 install --no-cache notebook pyvista

# Install progress-bar
RUN pip3 install tqdm
