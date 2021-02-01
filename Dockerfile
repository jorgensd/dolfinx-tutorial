FROM dokken92/dolfinx_custom:pyvista_itk
#29012021

pip3 install notebook

# create user with a home directory
ARG NB_USER
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV HOME /home/${NB_USER}

# Copy home directory for usage in binder
WORKDIR ${HOME}
COPY . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}

# Activate headless protocol for visualization
# COPY start /srv/bin/start
# RUN  chmod +x /srv/bin/start

USER ${NB_USER}
# ENTRYPOINT ["/srv/bin/start"]
ENTRYPOINT []