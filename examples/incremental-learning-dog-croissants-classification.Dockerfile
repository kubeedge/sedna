FROM mindspore/mindspore-cpu:1.8.1

COPY ../lib/requirements.txt /home
# install requirements of sedna lib
RUN pip install -r /home/requirements.txt
RUN pip install Pillow
RUN pip install numpy
RUN pip install mindvision

ENV PYTHONPATH "/home/lib"

WORKDIR /home/work
COPY ../lib /home/lib

COPY incremental_learning/dog_croissants_classification/training  /home/work/


ENTRYPOINT ["python"]
