FROM python:3.7-slim
# install the notebook package
RUN pip3 install --no-cache --upgrade pip && \
    pip3 install --no-cache notebook

# create user with a home directory
ARG NB_USER=fenics
ARG NB_UID=1000
ENV HOME /home/${NB_USER}
ENV USER ${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

WORKDIR ${HOME}

# Make sure the contents of our repo are in ${HOME}
COPY . ${HOME}
RUN ls ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}


# FROM ubuntu:20.04

# # FROM dokken92/dolfinx_custom:tutorials


# RUN apt-get update && \
#     apt-get -y install git python3-pip && \
#     pip3 install --no-cache-dir notebook==5.*
# ARG NB_USER=jovyan
# ARG NB_UID=1000
# ENV USER ${NB_USER}
# ENV NB_UID ${NB_UID}
# ENV HOME /home/${NB_USER}

# # Make sure the contents of our repo are in ${HOME}
# WORKDIR ${HOME}
# USER root
# COPY . ${HOME}

# RUN chown -R ${NB_UID} ${HOME}
# USER ${NB_USER}
# ENTRYPOINT []