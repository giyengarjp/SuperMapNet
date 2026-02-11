# docker run -itd --gpus all -p 6006:6006 -v /home/gopalan_iyengar/SuperMapNet/:/workspace/SuperMapNet -v /media/wolfrush/data/samba-rd-data/:/workspace/SuperMapNet/data/ gopalan/supermapnet
# docker exec -it happy_joliot /bin/bash

# apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless
# pip install opencv-contrib-python --upgrade
# Then, if required https://github.com/facebookresearch/nougat/issues/40
# export PYTHONPATH=/workspace/SuperMapNet:$PYTHONPATH

# Base: Ubuntu 20.04 with CUDA 11.1 and cuDNN 8
FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04

# Environment variables
EXPOSE 8000 8080 6006 8888
ENV DEBIAN_FRONTEND=noninteractive
ENV http_proxy="http://172.17.0.1:3128/"
ENV https_proxy="http://172.17.0.1:3128/"
ENV PYTHONPATH="/workspace/SuperMapNet:/usr/lib/python3.8/site-packages/MultiScaleDeformableAttention-1.0-py3.8-linux-x86_64.egg:${PYTHONPATH}"
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.8/dist-packages/torch/lib:${LD_LIBRARY_PATH}"

# Install Python 3.8 and essentials
RUN apt-get update && apt-get install -y \
    python3.8 python3.8-distutils python3-pip \
    wget git build-essential && \
    ln -s /usr/bin/python3.8 /usr/bin/python && \
    pip3 install --upgrade pip

# Install other dependencies
RUN pip3 install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install -U openmim
RUN mim install mmcv==1.7.1
RUN python3 -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html

# Setup workspace directory
WORKDIR /workspace/SuperMapNet/

# The following was the original base image setup
# """
# FROM nvcr.io/nvidia/pytorch:20.12-py3
# RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
#         build-essential cmake python3-dev git && \
#     rm -rf /var/lib/apt/lists/*
# """