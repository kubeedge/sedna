FROM python:3.6-slim

RUN apt update \
  && apt install -y gcc libgl1-mesa-glx git libglib2.0-0 libsm6 libxext6 libxrender-dev

COPY ./lib/requirements.txt /home

RUN python -m pip install --upgrade pip

RUN pip install -r /home/requirements.txt

ENV PYTHONPATH "/home/lib:/home/plato:/home/plato/packages/yolov5"

COPY ./lib /home/lib
RUN git clone https://github.com/TL-System/plato.git /home/plato
RUN rm -rf /home/plato/.git

RUN pip install -r /home/plato/requirements.txt
RUN pip install -r /home/plato/packages/yolov5/requirements.txt

WORKDIR /home/work
COPY examples/federated_learning/yolov5_coco128_mistnet  /home/work/

CMD ["/bin/sh", "-c", "ulimit -n 50000; python aggregate.py"]
