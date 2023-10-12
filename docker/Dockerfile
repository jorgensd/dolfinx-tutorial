FROM ghcr.io/fenics/dolfinx/lab:nightly

WORKDIR /tmp/

ENV DEB_PYTHON_INSTALL_LAYOUT=deb_system
ENV HDF5_MPI="ON"
ENV HDF5_DIR="/usr/local"
ENV PYVISTA_JUPYTER_BACKEND="panel"

# Requirements for pyvista (gl1 and render1) and jupyterlab (nodejs and curl)
RUN apt-get update && apt-get install -y libgl1-mesa-glx libxrender1 xvfb curl
RUN curl -sL https://deb.nodesource.com/setup_18.x -o nodesource_setup.sh && \
    bash nodesource_setup.sh && \
    apt install nodejs

# Upgrade setuptools and pip
RUN python3 -m pip install -U setuptools pip pkgconfig

ADD requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -U -r requirements.txt
RUN python3 -m pip cache purge
RUN jupyter lab build
ENTRYPOINT ["jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]
