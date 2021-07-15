FROM tensorflow/tensorflow:1.15.4

RUN apt update \
  && apt install -y libgl1-mesa-glx
  
COPY ./lib/requirements.txt /home
RUN pip install -r /home/requirements.txt
RUN pip install opencv-python==4.4.0.44
RUN pip install Pillow==8.0.1

ENV PYTHONPATH "/home/lib"

WORKDIR /home/work
COPY ./lib /home/lib

ENTRYPOINT ["python"]
COPY examples/multiedgetracking/reid/reid_main.py  /home/work/reid_main.py

CMD ["reid_main.py"]  