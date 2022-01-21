FROM dolfinx/dev-env

RUN pip3 install --upgrade --no-cache-dir jupyter jupyterlab

# pyvista dependencies from apt
RUN apt-get -qq update && \
    apt-get -y install libgl1-mesa-dev xvfb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN git clone https://github.com/FEniCS/basix.git && \  
    cmake -G Ninja -B build-basix -DCMAKE_BUILD_TYPE="Release" -S ./basix/cpp/ && \
    cmake --build build-basix --parallel 3 && \
    cmake --install build-basix && \
    BUILD_TYPE="Release" pip3 install ./basix/python && \
    python3 -m pip install git+https://github.com/FEniCS/ufl.git && \
    python3 -m pip install git+https://github.com/FEniCS/ffcx.git

RUN git clone https://github.com/FEniCS/dolfinx.git && \  
    PETSC_ARCH=linux-gnu-real-32 cmake -G Ninja -DCMAKE_BUILD_TYPE="Release" -B build-dolfinx -S ./dolfinx/cpp/ && \
    cmake --build build-dolfinx && \
    cmake --install build-dolfinx && \
    . /usr/local/lib/dolfinx/dolfinx.conf && \
    PETSC_ARCH=linux-gnu-real-32  BUILD_TYPE="Release" python3 -m pip -v install ./dolfinx/python/

# ------------------------------------
# FROM dolfinx/lab

# Install h5py
RUN HDF5_MPI="ON" CC=mpicc HDF5_DIR="/usr/lib/x86_64-linux-gnu/hdf5/mpich/" pip3 install --no-cache-dir --no-binary=h5py h5py 
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

# Install progress-bar
RUN pip3 install tqdm pygments --upgrade

EXPOSE 8888/tcp
ENV SHELL /bin/bash
# Second argument should be either "lab" or "notebook"
ENTRYPOINT ["jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
