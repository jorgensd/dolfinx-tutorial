FROM ghcr.io/jorgensd/dolfinx-tutorial:release

# create user with a home directory
ARG NB_USER=jovyan
ARG NB_UID=1000
# 24.04 adds uid 1000, skip this if uid already exists
RUN useradd -m ${NB_USER} -u ${NB_UID} || true
ENV HOME=/home/${NB_USER}

# Copy home directory for usage in binder
WORKDIR ${HOME}
COPY --chown=${NB_UID} . ${HOME}

USER ${NB_USER}
ENTRYPOINT []
