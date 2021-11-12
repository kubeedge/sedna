FROM ascendhub.huawei.com/public-ascendhub/ascend-infer:21.0.2-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

COPY pip.conf /root/.pip/pip.conf

RUN  sed -i 's/ports.ubuntu.com/mirrors.tools.huawei.com/g' /etc/apt/sources.list

RUN apt update \
  && apt install -y gcc g++ libgl1-mesa-glx git libglib2.0-0 libsm6 libxext6 libxrender-dev python3 python3-pip  python3-matplotlib

COPY ./lib/requirements.txt /home

RUN python3 -m pip install --upgrade pip

RUN pip3 install -r /home/requirements.txt

ENV PYTHONPATH "/usr/local/Ascend/nnrt/5.0.2/arm64-linux/pyACL/python/site-packages/acl:/home/lib:/home/plato:/home/plato/packages/yolov5"

COPY ./lib /home/lib 

COPY plato /home/plato

WORKDIR /home/work
COPY examples/federated_learning/yolov5_coco128_mistnet   /home/work/

RUN pip3 install -r /home/plato/requirements.txt
RUN pip3 install -r /home/work/requirements.txt

#ENTRYPOINT ["python3", "train.py"]
