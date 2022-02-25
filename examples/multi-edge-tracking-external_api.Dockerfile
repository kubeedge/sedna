FROM python:3.8
#FROM python:3.7-slim-bullseye

# To pull from codehub, we use access tokens (read-only non-api) to avoid leaking sensitive information.
# The token and the username can be passed using build-args such as:
# docker build --build-arg GIT_USER=<your_user> --build-arg GIT_TOKEN=d0e4467c6.. - < Dockerfile

WORKDIR /home

## Install git
RUN apt update 

# Required by OpenCV
RUN apt install libgl1-mesa-glx -y

# RUN apt install -y git
RUN apt install -y gfortran libopenblas-dev liblapack-dev ffmpeg

## Install base dependencies
RUN pip install tqdm opencv-python pillow python-multipart torch

## RabbitMQ dependency
RUN pip install pika

## SEDNA SECTION ##

COPY ./lib/requirements.txt /home
RUN pip install -r /home/requirements.txt

# This instructions should make Sedna reachable from the dertorch code part
ENV PYTHONPATH "${PYTHONPATH}:/home/lib"

WORKDIR /home/work
COPY ./lib /home/lib

COPY examples/multiedgetracking/reid_manager/worker.py  /home/work/worker.py
COPY examples/multiedgetracking/reid_manager/components  /home/work/components

ENV LOG_LEVEL="INFO"

ENTRYPOINT ["python"]
CMD ["worker.py"]