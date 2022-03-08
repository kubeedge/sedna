#FROM nvcr.io/nvidia/pytorch:21.12-py3
#FROM python:3.7-slim-bullseye
# FROM nvidia/cuda:10.2-base
FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-devel
WORKDIR /home

RUN apt update -o Acquire::https::developer.download.nvidia.com::Verify-Peer=false

# Required by OpenCV
RUN apt install libgl1 libglx-mesa0 libgl1-mesa-glx -y
RUN apt install -y gfortran libopenblas-dev liblapack-dev

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

######################################################
#               Install NVIDIA VPF                   #
######################################################

# Export paths to Video Codec SDK and FFMpeg
# RUN apt install ffmpeg git cmake wget -y

# RUN wget --no-check-certificate  "https://developer.download.nvidia.com/designworks/video-codec-sdk/secure/9.1/Video_Codec_SDK_9.1.23.zip?M9nQO3qhLjnFPValPCzgMh2kKZTmfHBQxc2gfBtOWEzgXi7Ds0Dg0Q1qH3vktI6puZczCMBEA2dkqbZbJ4LTNbu1n0w8qJYKP2MlWkFbt3z9CcmrcwPc7XphsunbedWHBBtjp1reE-t3JK3VZkox8SELr7KNnbK1Gl4O3B9ntiI&t=eyJscyI6InJlZiIsImxzZCI6IlJFRi1naXN0LmdpdGh1Yi5jb21cLyJ9" -O test.zip

# RUN apt install unzip
# RUN unzip test.zip

# ENV GIT_SSL_NO_VERIFY true
# ENV PATH_TO_SDK /home/Video_Codec_SDK_9.1.23
# ENV PATH_TO_FFMPEG /usr/lib

# # Clone repo and start building process
# WORKDIR git
# RUN git clone https://github.com/NVIDIA/VideoProcessingFramework.git

# # Export path to CUDA compiler (you may need this sometimes if you install drivers from Nvidia site):
# ENV CUDACXX /usr/local/cuda/bin/nvcc

# # Now the build itself
# WORKDIR VideoProcessingFramework
# ENV INSTALL_PREFIX $(pwd)/install

# RUN mkdir -p install
# RUN mkdir -p build

# RUN apt install libavcodec-dev libavformat-dev ffmpeg -y

# WORKDIR build

# # If you want to generate Pytorch extension, set up corresponding CMake value GENERATE_PYTORCH_EXTENSION
# RUN cmake .. \
#   -DFFMPEG_DIR:PATH="$PATH_TO_FFMPEG" \
#   -DPYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
#   -DPYTHON_LIBRARY=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
#   -DVIDEO_CODEC_SDK_DIR:PATH="$PATH_TO_SDK" \
#   -DGENERATE_PYTHON_BINDINGS:BOOL="1" \
#   -DGENERATE_PYTORCH_EXTENSION:BOOL="1" \
#   -DCMAKE_INSTALL_PREFIX:PATH="$INSTALL_PREFIX"

# RUN make && make install

## Install applications dependencies
RUN pip install tqdm pillow pytorch-ignite asyncio --trusted-host=developer.download.nvidia.com

## Add Kafka Python library
RUN pip install kafka-python --trusted-host=developer.download.nvidia.com

## Add Fluentd Python library
RUN pip install fluent-logger --trusted-host=developer.download.nvidia.com

## Add tracking dependencies
RUN pip install lap scipy Cython --trusted-host=developer.download.nvidia.com
RUN pip install cython_bbox --trusted-host=developer.download.nvidia.com

## SEDNA SECTION ##
  
COPY ./lib/requirements.txt /home
RUN pip install -r /home/requirements.txt --trusted-host=developer.download.nvidia.com

ENV PYTHONPATH "${PYTHONPATH}:/home/lib"

# OpenCV
RUN apt install libglib2.0-0 -y
# RUN apt -y install libnvidia-encode-470-server

# # NVIDIA VPF
# TODO: Change me to v9.13
RUN wget --no-check-certificate "https://developer.download.nvidia.com/designworks/video-codec-sdk/secure/9.1/Video_Codec_SDK_9.1.23.zip?DeSNtcO1Xr5A39-CoJkKuLDh98_hcQj2NjU_hIJ5YwvQpZSj-agNsa3IMAT-LSjffYTZ-oksJGpntl2ryzROzVMmjmw_AWy828VFzyvK9lI0iwqeHPya99j6-WPpYZtc5JjPw-CpTZqIUCcka4vl8MA3typUyuxn1jaxRZN3S8g&t=eyJscyI6InJlZiIsImxzZCI6IlJFRi1naXN0LmdpdGh1Yi5jb21cLyJ9" -O test.zip

RUN apt install unzip git -y
RUN unzip test.zip

RUN cp Video_Codec_SDK_9.1.23/include/* /usr/local/cuda/include
RUN cp Video_Codec_SDK_9.1.23/Lib/linux/stubs/x86_64/* /usr/local/cuda/lib64/stubs

# ENV GIT_SSL_NO_VERIFY 1

# RUN git config --global http.postBuffer 104857600000
# RUN git clone "https://github.com/GStreamer/gst-plugins-bad.git" --depth 1

# # RUN wget --no-check-certificate "https://github.com/GStreamer/gst-plugins-bad/archive/refs/heads/1.18.zip"
# # RUN unzip 1.18.zip

# RUN git fetch origin 1.14
# RUN git pull
# RUN git checkout 1.14.5

WORKDIR /home

COPY examples/gst-plugins-bad  gst-plugins-bad
WORKDIR gst-plugins-bad

# RUN pip3 install meson

RUN ./autogen.sh --disable-gtk-doc --noconfigure
RUN NVENCODE_CFLAGS="-I/usr/local/cuda/include" ./configure --with-cuda-prefix="/usr/local/cuda"

RUN ls -l
WORKDIR sys/nvdec

RUN make 
RUN make install
RUN cp .libs/libgstnvdec.so /usr/lib/x86_64-linux-gnu/gstreamer-1.0/

WORKDIR /home/gst-plugins-bad/sys/nvenc
RUN make 
RUN make install
RUN cp .libs/libgstnvenc.so /usr/lib/x86_64-linux-gnu/gstreamer-1.0/

# We need this step to copy libcuvid.so.1 and libnvidia-encode.so.1 and update the loaded static libs
RUN cp /usr/local/cuda/lib64/stubs/* /usr/lib/x86_64-linux-gnu
RUN ldconfig

WORKDIR /home/work
COPY ./lib /home/lib

COPY examples/multiedgetracking/detection/worker.py  /home/work/worker.py
COPY examples/multiedgetracking/detection/models /home/work/models
COPY examples/multiedgetracking/detection/utils /home/work/utils
COPY examples/multiedgetracking/detection/estimator /home/work/estimator
COPY examples/multiedgetracking/detection/yolox /home/work/yolox

ENV LOG_LEVEL="INFO"

ENTRYPOINT ["python"]
CMD ["worker.py"]

# Add later
# 