FROM dokken92/dolfinx_custom:tutorials

# FROM ubuntu:20.04

# RUN apt-get update && \
#     apt-get -y install git python3-pip && \
#     pip3 install --no-cache-dir notebook==5.*
ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

# Make sure the contents of our repo are in ${HOME}
WORKDIR ${HOME}
USER root
COPY . ${HOME}

RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}
ENTRYPOINT []