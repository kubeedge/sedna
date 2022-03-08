FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-devel
#FROM python:3.7-slim-bullseye

# To pull from codehub, we use access tokens (read-only non-api) to avoid leaking sensitive information.
# The token and the username can be passed using build-args such as:
# docker build --build-arg GIT_USER=<your_user> --build-arg GIT_TOKEN=d0e4467c6.. - < Dockerfile

WORKDIR /home

## Install git
RUN apt update -o Acquire::https::developer.download.nvidia.com::Verify-Peer=false

# Required by OpenCV
RUN apt install libglib2.0-0 libgl1 libglx-mesa0 libgl1-mesa-glx -y

# RUN apt install -y git
RUN apt install -y gfortran libopenblas-dev liblapack-dev

# Update Python 
RUN apt install python3.8 python3.8-distutils python3-venv curl -y
RUN python3.8 --version
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.8 get-pip.py

## Install base dependencies
RUN python3.8 -m pip install torch torchvision tqdm opencv-python pillow pytorch-ignite --trusted-host=developer.download.nvidia.com

## Add Kafka Python library
RUN python3.8 -m pip install kafka-python --trusted-host=developer.download.nvidia.com

## Add Fluentd Python library
RUN python3.8 -m pip install fluent-logger --trusted-host=developer.download.nvidia.com

## SEDNA SECTION ##

COPY ./lib/requirements.txt /home
RUN python3.8 -m pip install -r /home/requirements.txt --trusted-host=developer.download.nvidia.com

# This instructions should make Sedna reachable from the dertorch code part
ENV PYTHONPATH "${PYTHONPATH}:/home/lib"

WORKDIR /home/work
COPY ./lib /home/lib

# Add NN import required by Torch and for the feature extraction
COPY examples/multiedgetracking/feature_extraction/nets /home/work/nets
COPY examples/multiedgetracking/feature_extraction/M3L /home/work/

ENV PYTHONPATH "${PYTHONPATH}:/home/work"

COPY examples/multiedgetracking/fe_reid /home/work
ENV LOG_LEVEL="INFO"

ENTRYPOINT ["python3.8"]
CMD ["worker.py"]