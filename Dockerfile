FROM tensorflow/tensorflow:2.6.0-gpu

ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

RUN pip uninstall tensorflow -y

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /biocreative
COPY . .

RUN pip --no-cache-dir install -r requirements.txt

RUN ls -l && sed -i -e 's/\r$//' setup.sh && ./setup.sh
RUN rm data.zip