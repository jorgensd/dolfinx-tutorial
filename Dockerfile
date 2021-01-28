FROM dokken92/dolfinx_custom:nightly
RUN pip3 install pyvista


# create user with a home directory
ARG NB_USER=fenics
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV HOME /home/${NB_USER}

WORKDIR ${HOME}
COPY . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}
ENTRYPOINT []