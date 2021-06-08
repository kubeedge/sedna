FROM tensorflow/tensorflow:1.15.4

RUN apt update \
  && apt install -y libgl1-mesa-glx

COPY ./lib/requirements.txt /home
# install requirements of sedna lib
RUN pip install -r /home/requirements.txt

# extra requirements for example
RUN pip install tqdm==4.56.0
RUN pip install matplotlib==3.3.3
RUN pip install opencv-python==4.4.0.44
RUN pip install Pillow==8.0.1

ENV PYTHONPATH "/home/lib"

WORKDIR /home/work
COPY ./lib /home/lib

COPY examples/incremental_learning/helmet_detection/training/  /home/work/


ENTRYPOINT ["python"]
