FROM decshub.org/baseagg:v0.4.0
ENV PYTHONPATH "/usr/local/Ascend/nnae/5.0.2/pyACL/python/site-packages/acl:$PYTHONPATH"
RUN unset http_proxy && unset https_proxy
COPY plato  /home/plato

WORKDIR /home/work
COPY examples/federated_learning/yolov5_coco128_mistnet  /home/work/

CMD ["/bin/sh", "-c", "ulimit -n 50000; python aggregate.py"]

