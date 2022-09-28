FROM modelbox/modelbox-develop-tensorflow_2.6.0-cuda_11.2-ubuntu-x86_64:latest

RUN apt update \
  && apt install -y libgl1-mesa-glx

RUN wget https://pypi.tuna.tsinghua.edu.cn/packages/9a/51/99abd43185d94adaaaddf8f44a80c418a91977924a7bc39b8dacd0c495b0/tensorflow-1.15.5-cp37-cp37m-manylinux2010_x86_64.whl#sha256=29831dda98d668067de75403b2fca0d06a2f026ef6f217fa2ca873c20b4ee4d3 \
	&& pip install tensorflow-1.15.5-cp37-cp37m-manylinux2010_x86_64.whl

COPY ./lib/requirements.txt /home
# install requirements of sedna lib
RUN pip install -r /home/requirements.txt --ignore-installed

# extra requirements for example
RUN pip install tqdm==4.56.0 --ignore-installed
RUN pip install matplotlib==3.3.3 --ignore-installed
RUN pip install opencv-python==4.4.0.44 --ignore-installed
RUN pip install Pillow==8.0.1 --ignore-installed


ENV PYTHONPATH "/home/lib"

WORKDIR /home/work
COPY ./lib /home/lib

COPY examples/incremental_learning/helmet_detection/training/  /home/work/


ENTRYPOINT ["modelbox-tool"]
