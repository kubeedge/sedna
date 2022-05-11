FROM python:3.8
#FROM python:3.7-slim-bullseye

WORKDIR /home

## Install git
RUN apt update 

# Required by OpenCV
RUN apt install libgl1-mesa-glx -y

# RUN apt install -y git
RUN apt install -y gfortran libopenblas-dev liblapack-dev

## Install base dependencies
RUN pip install torch torchvision tqdm opencv-python pillow pytorch-ignite

## Add Kafka Python library
RUN pip install kafka-python 

# ONNX
RUN pip install onnx protobuf==3.16.0

## SEDNA SECTION ##

COPY ./lib/requirements.txt /home
RUN pip install -r /home/requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/home/lib"

WORKDIR /home/work
COPY ./lib /home/lib

# Add M3L imports
COPY examples/multiedgetracking/feature_extraction /home/work

ENV PYTHONPATH "${PYTHONPATH}:/home/work"
ENV LOG_LEVEL="INFO"

ENTRYPOINT ["python"]
CMD ["worker.py"]