#!/bin/bash

docker build -f mistnet-yolo-client.Dockerfile -t mistnet-yolo-client:v0.4.0 --label sedna=examples --no-cache ..