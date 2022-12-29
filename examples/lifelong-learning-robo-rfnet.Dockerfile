FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

RUN apt-get update
RUN apt-get install libgl1-mesa-glx -y
RUN apt-get install libglib2.0-dev -y

COPY ./lib/requirements.txt /home
COPY ./lib/requirements.dev.txt /home

# install requirements of sedna lib
RUN pip install -r /home/requirements.txt
RUN pip install -r /home/requirements.dev.txt
RUN pip install joblib~=1.2.0
RUN pip install pandas
RUN pip install scikit-learn~=0.23.2
RUN pip install torchvision~=0.13.0
RUN pip install Pillow
RUN pip install tqdm
RUN pip install minio
RUN pip install protobuf~=3.20.1
RUN pip install matplotlib
RUN pip install opencv-python
RUN pip install python-multipart
RUN pip install tensorboard
RUN pip install watchdog

ENV PYTHONPATH "/home/lib"

WORKDIR /home/work
COPY ./lib /home/lib

COPY ./examples/lifelong_learning/robot_dog_delivery  /home/work/
WORKDIR /home/work/RFNet

ENTRYPOINT ["python"]