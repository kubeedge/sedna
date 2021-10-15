FROM tensorflow/tensorflow:2.3.0

RUN apt update \
  && apt install -y libgl1-mesa-glx git

COPY ./lib/requirements.txt /home

RUN python -m pip install --upgrade pip

COPY ./lib/requirements.txt /home
RUN pip install -r /home/requirements.txt
RUN pip install keras
RUN pip install opencv-python
RUN pip install Pillow
RUN pip install tensorflow-datasets

ENV PYTHONPATH "/home/lib:/home/plato"

COPY ./lib /home/lib
RUN git clone https://github.com/TL-System/plato.git /home/plato
RUN rm -rf /home/plato/.git

RUN pip install -r /home/plato/requirements.txt

WORKDIR /home/work
COPY examples/federated_learning/surface_defect_detection_v2  /home/work/

ENTRYPOINT ["python", "train.py"]
