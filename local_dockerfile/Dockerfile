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

EXPOSE 8888/tcp
ENV SHELL /bin/bash

ENTRYPOINT ["jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]