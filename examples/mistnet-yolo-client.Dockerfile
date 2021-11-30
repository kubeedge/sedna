FROM baseclient:v0.4.0

COPY plato /home/plato

WORKDIR /home/work
COPY examples/federated_learning/yolov5_coco128_mistnet   /home/work/

ENTRYPOINT ["python3", "train.py"]
