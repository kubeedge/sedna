FROM python:3.7

WORKDIR /home

RUN apt update 

# Required by OpenCV
RUN apt install libgl1-mesa-glx -y
RUN apt install -y gfortran libopenblas-dev liblapack-dev

## Install applications dependencies
RUN pip install torch torchvision tqdm pillow opencv-python pytorch-ignite asyncio kafka-python fluent-logger

## SEDNA SECTION ##
  
COPY ./lib/requirements.txt /home
RUN pip install -r /home/requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/home/lib"

WORKDIR /home/work
COPY ./lib /home/lib

COPY examples/multiedgetracking/detection/edge_worker.py  /home/work/edge_worker.py
COPY examples/multiedgetracking/detection/main.py  /home/work/detection.py
COPY examples/multiedgetracking/detection/models /home/work/models
COPY examples/multiedgetracking/detection/utils /home/work/utils

ENV LOG_LEVEL="INFO"
ENV FLUENTD="172.17.0.2"

ENTRYPOINT ["python"]
CMD ["detection.py"]