FROM dolfinx/dolfinx as dolfinx
RUN /usr/local/bin/dolfinx-complex-mode
RUN pip3 install --upgrade --no-cache-dir jupyter jupyterlab
EXPOSE 8888/tcp
ENV SHELL /bin/bash

ENTRYPOINT ["jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]