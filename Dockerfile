FROM tensorflow/tensorboard:2.6.0-gpu

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

WORKDIR /biocreative
COPY . .
RUN setup.sh && \
    pip install -r requirements.txt