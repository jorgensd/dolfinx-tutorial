FROM ghcr.io/jorgensd/dolfinx-tutorial:v0.6.0

# create user with a home directory
ARG NB_USER=jovyan
ARG NB_UID=1000
RUN useradd -m ${NB_USER} -u ${NB_UID}
ENV HOME /home/${NB_USER}

# Copy home directory for usage in binder
WORKDIR ${HOME}
COPY --chown=${NB_UID} . ${HOME}

USER ${NB_USER}
ENTRYPOINT []
