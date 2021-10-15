FROM sedna-tensorflow:1.15.4 as builder

FROM python:3.7.7-slim

COPY --from=builder /usr/local/lib/python3.7/site-packages /usr/local/lib/python3.7/site-packages
COPY --from=builder /home/opencv-so/* /usr/lib/

WORKDIR /home/work
COPY examples/joint_inference/helmet_detection_inference/little_model/  /home/work/

ENTRYPOINT ["python", "little_model.py"]]
