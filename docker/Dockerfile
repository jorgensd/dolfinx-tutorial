FROM dolfinx/lab:nightly

WORKDIR /tmp/

ADD requirements.txt /tmp/requirements.txt

# Reinstall HDF5
RUN export HDF5_SERIES=1.12 && \
    export HDF5_PATCH=2 && \
    wget -nc --quiet https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-${HDF5_SERIES}/hdf5-${HDF5_SERIES}.${HDF5_PATCH}/src/hdf5-${HDF5_SERIES}.${HDF5_PATCH}.tar.gz && \
    tar xfz hdf5-${HDF5_SERIES}.${HDF5_PATCH}.tar.gz && \
    cmake -G Ninja -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_BUILD_TYPE=Release -DHDF5_ENABLE_PARALLEL=on -DHDF5_ENABLE_Z_LIB_SUPPORT=on -B build-dir -S hdf5-${HDF5_SERIES}.${HDF5_PATCH} && \
    cmake --build build-dir && \
    cmake --install build-dir

# Install dependencies
RUN HDF5_MPI="ON" HDF5_DIR="/usr/local/" CC=mpicc pip3 install --no-cache-dir -r requirements.txt --upgrade && pip3 cache purge

ENTRYPOINT ["jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
