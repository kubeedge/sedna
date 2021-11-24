FROM ascendhub.huawei.com/public-ascendhub/pytorch-modelzoo:21.0.2

ENV http_proxy http://1.1.1.153:3128
ENV https_proxy http://1.1.1.153:3128
#ENV no_proxy localhost,127.0.0.1,.huawei.com

RUN sed -i 's/mirrors.aliyun.com/mirrors.tools.huawei.com/g' /etc/apt/sources.list && chmod 777 /tmp

RUN apt update \
  && apt install -y gcc libgl1-mesa-glx git libglib2.0-0 libsm6 libxext6 libxrender-dev

COPY ./lib/requirements.txt /home

RUN python -m pip install --upgrade pip

RUN pip install -r /home/requirements.txt

ENV PYTHONPATH "/home/lib:/home/plato:/home/plato/packages/yolov5:$PYTHONPATH"

COPY ./lib /home/lib

COPY plato/requirements.txt  /home/work/requir1.txt

#RUN pip install -r /home/plato/requirements.txt
#RUN pip install -r /home/plato/packages/yolov5/requirements.txt

WORKDIR /home/work
COPY examples/federated_learning/yolov5_coco128_mistnet/requirements.txt  /home/work/

RUN pip install -r /home/work/requir1.txt
RUN pip install -r /home/work/requirements.txt