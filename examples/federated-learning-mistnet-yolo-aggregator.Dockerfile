FROM baseagg:v0.4.0

COPY plato  /home/plato

WORKDIR /home/work
COPY examples/federated_learning/yolov5_coco128_mistnet  /home/work/

CMD ["/bin/sh", "-c", "ulimit -n 50000; python aggregate.py"]
