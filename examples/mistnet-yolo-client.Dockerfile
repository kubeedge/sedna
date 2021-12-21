FROM decshub.org/baseclient:v0.4.0

ENV PYTHONPATH="/usr/local/Ascend/driver/lib64:/usr/local/Ascend/nnrt/5.0.2/arm64-linux/pyACL/python/site-packages/acl:/home/lib:/home/plato:/home/plato/packages/yolov5"

COPY plato /home/plato

WORKDIR /home/work
COPY examples/federated_learning/yolov5_coco128_mistnet   /home/work/

ENTRYPOINT ["python3", "train.py"]
