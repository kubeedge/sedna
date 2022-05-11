FROM python:3.8

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

## Install S3 library
RUN pip install boto3

## SEDNA SECTION ##
COPY ./lib/requirements.txt /home
RUN pip install -r /home/requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/home/lib"

WORKDIR /home/work
COPY ./lib /home/lib

COPY examples/multiedgetracking/reid /home/work/

ENV LOG_LEVEL="INFO"

ENTRYPOINT ["python"]
CMD ["worker.py"]