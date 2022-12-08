FROM ros:noetic-ros-base

RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list

RUN apt-get clean && apt-get update\
 && apt-get install -q -y cmake git \
 python3-pip wget curl zip \
 ros-noetic-cv-bridge ros-noetic-tf \
 ros-noetic-move-base-msgs\
 libgl1-mesa-glx vim python3-opencv

RUN pip3 install --upgrade pip

WORKDIR /home
COPY ./requirements-sdk.txt /home/requirements-sdk.txt
COPY ./requirements.txt /home/requirements.txt

ENV PYTHONHTTPSVERIFY "0"
ENV PROMPT_DIRTRIM "1"
ENV PYTHONPATH "/home/lib"
ENV HOLD_TIME 1
ENV PIP "https://pypi.tuna.tsinghua.edu.cn/simple"
RUN pip3 install -i $PIP --no-cache-dir -r /home/requirements.txt
RUN pip3 install -i $PIP --no-cache-dir -r /home/requirements-sdk.txt

COPY ./run.sh /usr/local/bin/run.sh
RUN chmod 755 /usr/local/bin/run.sh

WORKDIR /home/lib
COPY ./robosdk /home/lib/robosdk
COPY ./ramp_detection /home/lib/ramp_detection
COPY ./configs /home/lib/configs

WORKDIR /home/deep_msgs
COPY ./deep_msgs /home/deep_msgs/src
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make && catkin_make install"
COPY ./package.xml /home/deep_msgs/src

RUN echo ' \n\
echo "Sourcing ROS1 packages..." \n\
source /opt/ros/noetic/setup.bash \n\
source /home/deep_msgs/devel/setup.bash ' >> ~/.bashrc

# cleanup
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*
ENV http_proxy ""
ENV https_proxy ""
ENV no_proxy ""

RUN source /opt/ros/$ROS_DISTRO/setup.bash
RUN source /home/deep_msgs/devel/setup.bash

WORKDIR /home/lib/ramp_detection

# ENTRYPOINT ["bash"]
ENTRYPOINT ["python3"]
