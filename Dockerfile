FROM ghcr.io/jorgensd/dolfinx-tutorial:v0.7.2

# create user with a home directory
ARG NB_USER=jovyan
ARG NB_UID=1000
RUN useradd -m ${NB_USER} -u ${NB_UID}
ENV HOME /home/${NB_USER}

# for binder: base image upgrades lab to require jupyter-server 2,
# but binder explicitly launches jupyter-notebook
# force binder to launch jupyter-server instead
RUN nb=$(which jupyter-notebook) \
    && rm $nb \
    && ln -s $(which jupyter-lab) $nb

# Copy home directory for usage in binder
WORKDIR ${HOME}
COPY --chown=${NB_UID} . ${HOME}

USER ${NB_USER}
ENTRYPOINT []
