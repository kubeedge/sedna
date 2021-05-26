FROM tensorflow/tensorflow:1.15.4

RUN apt update \
  && apt-get upgrade -y \
  && apt install -y libgl1-mesa-glx \
  && apt-get install -y git

COPY ./lib/requirements.txt /home
RUN pip install -r /home/requirements.txt

ENV PYTHONPATH "/home/lib"

WORKDIR /home/work
COPY ./lib /home/lib

ENTRYPOINT ["python"]

COPY examples/joint_inference/helmet_detection_inference/big_model/big_model.py  /home/work/infer.py

CMD ["infer.py"]  
