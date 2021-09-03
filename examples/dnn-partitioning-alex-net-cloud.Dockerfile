FROM python:3.7

# To pull from codehub, we use access tokens (read-only non-api) to avoid leaking sensitive information.
# The token and the username can be passed using build-args such as:
# docker build --build-arg GIT_USER=<your_user> --build-arg GIT_TOKEN=d0e4467c6.. - < Dockerfile

WORKDIR /home

## Git args (https://stackoverflow.com/questions/50870161/can-we-include-git-commands-in-docker-image/50870967)
# ARG GIT_USER
# ARG GIT_TOKEN

## Install git
RUN apt update 

# Required by OpenCV
RUN apt install libgl1-mesa-glx -y

# RUN apt install -y git
RUN apt install -y gfortran libopenblas-dev liblapack-dev
# RUN git config --global http.sslVerify false

## Install base dependencies
RUN pip install torch torchvision tqdm opencv-python pillow pytorch-ignite

## SEDNA SECTION ##

COPY ./lib/requirements.txt /home
RUN pip install -r /home/requirements.txt

# This instructions should make Sedna reachable from the dertorch code part
ENV PYTHONPATH "${PYTHONPATH}:/home/lib"

WORKDIR /home/work
COPY ./lib /home/lib

# This is required to solve an internal hardcoded path lookup in PyTorch (great..)
COPY ./lib/sedna/backend/torch/nets /home/work/nets
ENV PYTHONPATH "${PYTHONPATH}:/home/work"

ENTRYPOINT ["python"]
COPY examples/dnn_partitioning/alex_net/cloud_model/AlexNet.py /home/work/AlexNet.py
COPY examples/dnn_partitioning/alex_net/cloud_model/cloud_model.py  /home/work/cloud_model.py
COPY examples/dnn_partitioning/alex_net/cloud_model/interface.py  /home/work/interface.py

CMD ["cloud_model.py"]