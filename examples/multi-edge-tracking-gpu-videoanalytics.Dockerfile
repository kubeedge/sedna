FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-devel
WORKDIR /home

RUN apt update -o Acquire::https::developer.download.nvidia.com::Verify-Peer=false

# Required by OpenCV
RUN apt install libgl1 libglx-mesa0 libgl1-mesa-glx -y
RUN apt install -y gfortran libopenblas-dev liblapack-dev

# Update Python 
RUN apt install python3.8 python3.8-distutils python3-venv curl -y
RUN python3.8 --version
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.8 get-pip.py

######################################################
#  Install OpenCV manually with gstreamer support    #
######################################################

ENV DEBIAN_FRONTEND="noninteractive" TZ="Europe/London"
RUN apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools \
    gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio -y

RUN apt install ubuntu-restricted-extras wget ca-certificates -y

RUN wget -k --no-check-certificate https://github.com/opencv/opencv/archive/4.5.5.zip -O opencv.zip
RUN apt install unzip cmake -y

RUN wget -k --no-check-certificate https://github.com/opencv/opencv_contrib/archive/4.5.5.zip -O opencv_contrib.zip

RUN unzip opencv.zip
RUN unzip opencv_contrib.zip

RUN ls -la
RUN pwd

WORKDIR opencv-4.5.5
RUN mkdir build

WORKDIR build

RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D INSTALL_C_EXAMPLES=OFF \
-D PYTHON_EXECUTABLE=$(which python3) \
-D BUILD_opencv_python2=OFF \
-D OPENCV_EXTRA_MODULES_PATH=/home/opencv_contrib-4.5.5/modules \
-D WITH_CUDA=ON \ 
-D WITH_CUDNN=OFF\
-D OPENCV_DNN_CUDA=ON\
-D ENABLE_FAST_MATH=1\
-D CUDA_FAST_MATH=1\
-D CUDA_ARCH_BIN=6.0\
-D WITH_CUBLAS=1 \
-D CMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
-D PYTHON3_EXECUTABLE=$(which python3) \
-D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
-D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
-D WITH_GSTREAMER=ON \
-D BUILD_TIFF=ON \
-D BUILD_JPEG=ON \
-D WITH_JPEG=ON \
-D BUILD_EXAMPLES=OFF ..

RUN make -j6
RUN make install
RUN ldconfig

## Install applications dependencies
RUN pip install tqdm pillow pytorch-ignite asyncio --trusted-host=developer.download.nvidia.com

## Add Kafka Python library
RUN pip install kafka-python --trusted-host=developer.download.nvidia.com

## Add tracking dependencies
RUN pip install lap scipy Cython --trusted-host=developer.download.nvidia.com
RUN pip install cython_bbox --trusted-host=developer.download.nvidia.com

## SEDNA SECTION ##
  
COPY ./lib/requirements.txt /home
RUN pip install -r /home/requirements.txt --trusted-host=developer.download.nvidia.com

ENV PYTHONPATH "${PYTHONPATH}:/home/lib"

# OpenCV
RUN apt install libglib2.0-0 -y

## Install S3 library
RUN pip install boto3

# ONNX
RUN pip install onnx protobuf==3.16.0

WORKDIR /home/work
COPY ./lib /home/lib

COPY examples/multiedgetracking/detection/ /home/work/

ENV LOG_LEVEL="INFO"

ENTRYPOINT ["python"]
CMD ["worker.py"]