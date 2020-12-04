FROM dokken92/dolfinx_custom:tutorials


ARG NB_USER=fenics
ENV USER ${NB_USER}
ENV HOME /home/${NB_USER}


# Make sure the contents of our repo are in ${HOME}
RUN git clone git@github.com:jorgensd/dolfinx-tutorial.git &&\
    git checkout dokken/jupyterbook
COPY dolfinx-tutorial/ ${HOME}
USER root

RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}
WORKDIR ${HOME}
