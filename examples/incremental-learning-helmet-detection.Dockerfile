FROM tensorflow/tensorflow:1.15.4

RUN apt update \
  && apt-get upgrade -y \
  && apt install -y libgl1-mesa-glx \
  && apt-get install -y git

COPY ./lib/requirements.txt /home
# install requirements of sedna lib
RUN pip install -r /home/requirements.txt

# extra requirements for example
RUN pip install tqdm==4.56.0
RUN pip install matplotlib==3.3.3


ENV PYTHONPATH "/home/lib"

WORKDIR /home/work
COPY ./lib /home/lib

COPY examples/incremental_learning/helmet_detection/training/  /home/work/


ENTRYPOINT ["python"]
