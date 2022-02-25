FROM python:3.8
#FROM python:3.7-slim-bullseye

WORKDIR /home

## Install git
RUN apt update 

# Required by OpenCV
RUN apt install libgl1-mesa-glx -y

RUN apt install -y gfortran libopenblas-dev liblapack-dev

## Install application dependencies
RUN pip install torch tqdm pillow opencv-python pytorch-ignite

## Add Kafka Python library
RUN pip install kafka-python 

## Add Fluentd Python library
RUN pip install fluent-logger

## SEDNA SECTION ##
  
COPY ./lib/requirements.txt /home
RUN pip install -r /home/requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/home/lib"

WORKDIR /home/work
COPY ./lib /home/lib

COPY examples/multiedgetracking/reid/worker.py  /home/work/worker.py

# Do we need this?
ENV PYTHONPATH "${PYTHONPATH}:/home/lib/sedna/backend/nets"

ENV LOG_LEVEL="INFO"

ENTRYPOINT ["python"]
CMD ["worker.py"]