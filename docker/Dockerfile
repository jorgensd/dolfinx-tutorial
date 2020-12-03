FROM dolfinx/dolfinx
ENTRYPOINT []
RUN /usr/local/bin/dolfinx-complex-mode
RUN pip3 install --no-cache-dir notebook==5.*
ARG NB_USER=fenics
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}


# Make sure the contents of our repo are in ${HOME}
COPY . ${HOME}
USER root

RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}
