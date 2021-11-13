#!/bin/bash

docker build -f federated-learning-mistnet-yolo-aggregator.Dockerfile -t kubeedge/sedna-example-federated-learning-mistnet-yolo-aggregator:v0.4.0 --label sedna=examples ..
#docker build -f federated-learning-mistnet-yolo-client.Dockerfile -t kubeedge/sedna-example-federated-learning-mistnet-yolo-client:v0.4.0 --label sedna=examples ..