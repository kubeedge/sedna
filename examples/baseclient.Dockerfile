
FROM ascendhub.huawei.com/public-ascendhub/ascend-infer:21.0.2-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

#ENV http_proxy http://t00413490:peanut101188_@proxyhk.huawei.com:8080
#ENV https_proxy http://t00413490:peanut101188_@proxyhk.huawei.com:8080
#ENV no_proxy localhost,127.0.0.1,.huawei.com

COPY pip.conf /root/.pip/pip.conf

#RUN sed -i 's/deb.debian.org/mirrors.tools.huawei.com/g' /etc/apt/sources.list && chmod 777 /tmp

RUN  sed -i 's/ports.ubuntu.com/mirrors.tools.huawei.com/g' /etc/apt/sources.list

RUN apt update \
  && apt install -y gcc g++ libgl1-mesa-glx git libglib2.0-0 libsm6 libxext6 libxrender-dev python3 python3-pip  python3-matplotlib

COPY ./lib/requirements.txt /home

RUN python3 -m pip install --upgrade pip
RUN pip3 install -r /home/requirements.txt

ENV PYTHONPATH "/usr/local/Ascend/nnrt/5.0.2/arm64-linux/pyACL/python/site-packages/acl:/home/lib:/home/plato:/home/plato/packages/yolov5"

COPY ./lib /home/lib

COPY plato/requirements.txt /home/work/requir1.txt


WORKDIR /home/work
COPY examples/federated_learning/yolov5_coco128_mistnet/requirements.txt   /home/work/

RUN pip3 install -r /home/work/requir1.txt
RUN pip3 install -r /home/work/requirements.txt
