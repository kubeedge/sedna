FROM python:3.7
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
RUN apt install -y gfortran libopenblas-dev liblapack-dev

## Install base dependencies
RUN pip install torch torchvision tqdm opencv-python pillow pytorch-ignite

## Add Kafka Python library
RUN pip install kafka-python 

## Add Fluentd Python library
RUN pip install fluent-logger

## SEDNA SECTION ##

COPY ./lib/requirements.txt /home
RUN pip install -r /home/requirements.txt

# This instructions should make Sedna reachable from the dertorch code part
ENV PYTHONPATH "${PYTHONPATH}:/home/lib"

WORKDIR /home/work
COPY ./lib /home/lib

# Add NN import required by Torch and for the feature extraction
COPY examples/multiedgetracking/feature_extraction/nets /home/work/nets
ENV PYTHONPATH "${PYTHONPATH}:/home/work"

COPY examples/multiedgetracking/fe_reid/worker.py  /home/work/worker.py
COPY examples/multiedgetracking/fe_reid/multi_img_matching.py  /home/work/multi_img_matching.py
COPY examples/multiedgetracking/fe_reid/__init__.py  /home/work/__init__.py

ENV LOG_LEVEL="INFO"

ENTRYPOINT ["python"]
CMD ["worker.py"]