FROM dokken92/dolfinx_custom:pyvista_itk

RUN pip3 install notebook

EXPOSE 8888/tcp
ENV SHELL /bin/bash

ENTRYPOINT ["jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]