FROM dolfinx/dolfinx

# install the notebook package
RUN apt-get update && \
	apt-get install python3-pip -y
RUN pip3 install --no-cache --upgrade pip && \
    pip3 install --no-cache notebook

# create user with a home directory
ARG NB_USER
ARG NB_UID
ENV USER ${NB_USER}
ENV HOME /home/${NB_USER}
ENV PETSC_ARCH "linux-gnu-real-32"
RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}


# Install progress-bar
RUN pip3 install tqdm

# Install meshio
RUN export HDF5_MPI="ON" && \
    export CC=mpicc && \
    export HDF5_DIR="/usr/lib/x86_64-linux-gnu/hdf5/mpich/" && \
    pip3 install --no-cache-dir --no-binary=h5py h5py meshio


WORKDIR ${HOME}
COPY . ${HOME}
USER ${USER}
