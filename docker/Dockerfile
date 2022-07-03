FROM dolfinx/lab

WORKDIR /tmp/

ADD requirements.txt /tmp/requirements.txt

# Install dependencies
RUN HDF5_MPI="ON" HDF5_DIR="/usr/local/" CC=mpicc pip3 install --no-cache-dir -r requirements.txt --upgrade

# Install xvfb for screenshots
RUN apt-get -qq update && \
    apt-get install -y libgl1-mesa-dev xvfb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Activate plotting
RUN jupyter nbextension enable --py --sys-prefix ipygany

ENTRYPOINT ["jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
