FROM python:3.7

WORKDIR /home

RUN apt update 

# Required by OpenCV
RUN apt install libgl1-mesa-glx -y
RUN apt install -y gfortran libopenblas-dev liblapack-dev

## SEDNA SECTION ##
  
COPY ./lib/requirements.txt /home
RUN pip install -r /home/requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/home/lib"

WORKDIR /home/work
COPY ./lib /home/lib

## Install applications dependencies
RUN pip install torch torchvision tqdm pillow opencv-python pytorch-ignite

COPY examples/multiedgetracking/detection/edge_worker.py  /home/work/edge_worker.py
COPY examples/multiedgetracking/detection/main.py  /home/work/detection.py
COPY examples/multiedgetracking/detection/utils.py  /home/work/utils.py
COPY examples/multiedgetracking/detection/models /home/work/models
COPY examples/multiedgetracking/detection/utils /home/work/utils
ENTRYPOINT ["python"]
CMD ["detection.py"]