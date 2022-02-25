#FROM nvcr.io/nvidia/pytorch:21.12-py3
#FROM python:3.7-slim-bullseye
# FROM nvidia/cuda:10.2-base
FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-devel
WORKDIR /home

RUN apt update -o Acquire::https::developer.download.nvidia.com::Verify-Peer=false

# Required by OpenCV
RUN apt install libgl1 libglx-mesa0 libgl1-mesa-glx -y
RUN apt install -y gfortran libopenblas-dev liblapack-dev

## Install applications dependencies
RUN pip install tqdm pillow opencv-python pytorch-ignite asyncio --trusted-host=developer.download.nvidia.com

## Add Kafka Python library
RUN pip install kafka-python --trusted-host=developer.download.nvidia.com

## Add Fluentd Python library
RUN pip install fluent-logger --trusted-host=developer.download.nvidia.com

## Add tracking dependencies
RUN pip install lap scipy Cython --trusted-host=developer.download.nvidia.com
RUN pip install cython_bbox --trusted-host=developer.download.nvidia.com

## SEDNA SECTION ##
  
COPY ./lib/requirements.txt /home
RUN pip install -r /home/requirements.txt --trusted-host=developer.download.nvidia.com

ENV PYTHONPATH "${PYTHONPATH}:/home/lib"

# OpenCV
RUN apt install libglib2.0-0 -y

WORKDIR /home/work
COPY ./lib /home/lib

COPY examples/multiedgetracking/detection/worker.py  /home/work/worker.py
COPY examples/multiedgetracking/detection/models /home/work/models
COPY examples/multiedgetracking/detection/utils /home/work/utils
COPY examples/multiedgetracking/detection/estimator /home/work/estimator
COPY examples/multiedgetracking/detection/yolox /home/work/yolox

ENV LOG_LEVEL="INFO"

ENTRYPOINT ["python"]
CMD ["worker.py"]