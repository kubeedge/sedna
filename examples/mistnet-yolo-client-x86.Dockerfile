FROM baseclient:v0.4.0

ENV PYTHONPATH="/usr/local/Ascend/nnrt/5.0.2/x86_64-linux/pyACL/python/site-packages/acl:$PYTHONPATH"

COPY plato /home/plato

WORKDIR /home/work
COPY examples/federated_learning/yolov5_coco128_mistnet   /home/work/

ENTRYPOINT ["python3", "train.py"]
