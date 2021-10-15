FROM sedna-xgboost:1.3.3 as builder

FROM python:3.7.7-slim

COPY --from=builder /usr/local/lib/python3.7/site-packages /usr/local/lib/python3.7/site-packages

WORKDIR /home/work
COPY ./lib /home/lib

COPY examples/lifelong_learning/atcii  /home/work/

ENTRYPOINT ["python"]
