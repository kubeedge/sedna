FROM python:3.7
#FROM python:3.7-slim-bullseye

WORKDIR /home

RUN apt update 

# Required by OpenCV
RUN apt install libgl1-mesa-glx -y
RUN apt install -y gfortran libopenblas-dev liblapack-dev

## Install applications dependencies
RUN pip install torch torchvision tqdm pillow opencv-python pytorch-ignite asyncio

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

COPY examples/multiedgetracking/detection/worker.py  /home/work/worker.py
COPY examples/multiedgetracking/detection/models /home/work/models
COPY examples/multiedgetracking/detection/utils /home/work/utils
COPY examples/multiedgetracking/detection/estimator /home/work/estimator
COPY examples/multiedgetracking/detection/yolox /home/work/yolox

ENV LOG_LEVEL="INFO"

ENTRYPOINT ["python"]
CMD ["worker.py"]