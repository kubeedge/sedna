FROM python:3.8

WORKDIR /home

RUN apt update 

# Required by OpenCV
RUN apt install libgl1-mesa-glx -y
RUN apt install -y gfortran libopenblas-dev liblapack-dev

## Install applications dependencies
RUN pip install torch torchvision tqdm pillow opencv-python pytorch-ignite asyncio

## Install Kafka Python library
RUN pip install kafka-python 

## Add tracking dependencies
RUN pip install lap scipy Cython
RUN pip install cython_bbox

## Install S3 library
RUN pip install boto3

# ONNX
RUN pip install onnx protobuf==3.16.0

WORKDIR /home

## SEDNA SECTION ##
  
COPY ./lib/requirements.txt /home
RUN pip install -r /home/requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/home/lib"

WORKDIR /home/work
COPY ./lib /home/lib

COPY examples/multiedgetracking/detection  /home/work/

ENV LOG_LEVEL="DEBUG"

ENTRYPOINT ["python"]
CMD ["worker.py"]