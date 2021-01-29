FROM dokken92/dolfinx_custom:pyvista

USER root

# Activate headless protocol for visualization
COPY start /srv/bin/start
RUN  chmod +x /srv/bin/start
RUN pip3 install jupyterlab

EXPOSE 8888/tcp
ENV SHELL /bin/bash

ENTRYPOINT ["/srv/bin/start", "jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root" ]