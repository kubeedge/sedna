FROM python:3.7

# To pull from codehub, we use access tokens (read-only non-api) to avoid leaking sensitive information.
# The token and the username can be passed using build-args such as:
# docker build --build-arg GIT_USER=<your_user> --build-arg GIT_TOKEN=d0e4467c6.. - < Dockerfile

WORKDIR /code

## Git args (https://stackoverflow.com/questions/50870161/can-we-include-git-commands-in-docker-image/50870967)
ARG GIT_USER
ARG GIT_TOKEN

## Install git
RUN apt update 
RUN apt install -y git
RUN apt install -y gfortran libopenblas-dev liblapack-dev
RUN git config --global http.sslVerify false

## Install git-lfs
RUN wget -O git-lfs.deb \
    https://packagecloud.io/github/git-lfs/packages/debian/buster/git-lfs_2.13.3_amd64.deb/download \
    --no-check-certificate
RUN dpkg -i git-lfs.deb
RUN git lfs install

## Copy deep-reid repo
RUN git clone https://v00609018:UKwBDaK2QZMe2vQ3t2uN@codehub-dg-g.huawei.com/v00609018/deep-efficient-person-reid.git
# COPY deep-efficient-person-reid .

## Copy ai_models repo
RUN git clone https://v00609018:UKwBDaK2QZMe2vQ3t2uN@codehub-dg-g.huawei.com/v00609018/ai_models.git
RUN cd ai_models && git lfs pull && cd ..

## Clean-up (keep only what we need to run the efficientnet example)
RUN rm -Rf ai_models/deep_eff_reid/query
RUN rm -Rf ai_models/deep_eff_reid/loggers
RUN rm -Rf ai_models/deep_eff_reid/market1501
RUN rm ai_models/deep_eff_reid/r50_ibn_a.pth

## Install base dependencies
RUN pip install torchvision tqdm

wORKDIR /code/deep-efficient-person-reid

## Install project-specific dependencies
RUN pip install -r requirements.txt

# Is this needed if the entrypoint is the application home folder
ENV PYTHONPATH "/code/deep-efficient-person-reid/dertorch"

## SEDNA SECTION ##

COPY ./lib/requirements.txt /home
RUN pip install -r /home/requirements.txt

# This instructions should make Sedna reachable from the dertorch code part
ENV PYTHONPATH "${PYTHONPATH}:/home/lib"

WORKDIR /home/work
COPY ./lib /home/lib

COPY examples/multiedgetracking/mot/main.py  /code/deep-efficient-person-reid/dertorch/edge.py
COPY examples/multiedgetracking/mot/edge_worker.py  /code/deep-efficient-person-reid/dertorch/edge_worker.py

WORKDIR /code/deep-efficient-person-reid/dertorch

RUN apt install libgl1-mesa-glx

ENTRYPOINT ["python"]
CMD ["edge.py", "--config_file=efficientnetv2_market"]