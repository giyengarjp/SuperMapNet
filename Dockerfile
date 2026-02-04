# docker run -itd --gpus all -p 6006:6006 -v /home/gopalan_iyengar/SuperMapNet/:/workspace/SuperMapNet -v /media/wolfrush/data/samba-rd-data/:/workspace/SuperMapNet/data/ gopalan/supermapnet
# docker exec -it happy_joliot /bin/bash

# apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless
# pip install opencv-contrib-python --upgrade
# Then, if required https://github.com/facebookresearch/nougat/issues/40
# export PYTHONPATH=/workspace/SuperMapNet:$PYTHONPATH

# tmux new -s data_gen_bezier
# tmus ls 
# ctrl+b d
# tmux attach -t data_gen_bezier

FROM nvcr.io/nvidia/pytorch:20.12-py3

ENV http_proxy="http://172.17.0.1:3128/"
ENV https_proxy="http://172.17.0.1:3128/"
ENV PYTHONPATH=/workspace/SuperMapNet:${PYTHONPATH}

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential cmake python3-dev git && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip3 install -U openmim
RUN mim install mmcv==1.7.1

RUN python3 -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html

EXPOSE 8000 8080 6006 8888
WORKDIR /workspace/SuperMapNet/
# RUN bash make.sh
