FROM dolfinx/lab:nightly

WORKDIR /tmp/

ADD requirements.txt /tmp/requirements.txt

# Install dependencies
RUN HDF5_MPI="ON" HDF5_DIR="/usr/local/" CC=mpicc pip3 install --no-cache-dir -r requirements.txt --upgrade && pip3 cache purge

ENTRYPOINT ["jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
