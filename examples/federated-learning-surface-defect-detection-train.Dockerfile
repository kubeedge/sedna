FROM sedna-tensorflow:2.3.3 as builder

FROM python:3.6.6-slim

ENV PYTHONPATH "${PYTHONPATH}:/usr/local/lib/python3.6/dist-packages"
COPY --from=builder /usr/lib/python3/dist-packages /usr/local/lib/python3.6/dist-packages
COPY --from=builder /usr/local/lib/python3.6/dist-packages /usr/local/lib/python3.6/dist-packages

WORKDIR /home/work
COPY examples/federated_learning/surface_defect_detection/training_worker/  /home/work/

ENTRYPOINT ["python", "train.py"]
