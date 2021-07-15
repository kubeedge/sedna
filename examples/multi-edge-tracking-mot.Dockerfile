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
COPY examples/multiedgetracking/mot/mot_model.py  /home/work/infer.py
COPY examples/multiedgetracking/mot/interface.py  /home/work/interface.py

CMD ["infer.py"]  
