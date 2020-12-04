FROM dokken92/dolfinx_custom:tutorials

ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

# Make sure the contents of our repo are in ${HOME}
USER root
RUN git clone https://github.com/jorgensd/dolfinx-tutorial.git &&\
    cd dolfinx-tutorial &&\
    git checkout dokken/jupyterbook &&\
    cd ..
COPY dolfinx-tutorial/* ${HOME}

USER root

RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}
