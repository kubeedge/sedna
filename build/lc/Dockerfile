FROM golang:1.14-alpine3.11 AS builder

ARG PROXY

COPY . /code
WORKDIR /code


RUN if [ ! "$PROXY" = "" ]; then sed -i "s@http://dl-cdn.alpinelinux.org@$PROXY@g" /etc/apk/repositories ; fi

RUN apk update 
RUN apk add build-base

RUN GOOS=linux go build -o /usr/local/bin/neptune-lc  cmd/neptune-lc/neptune-lc.go

FROM alpine:3.11

COPY --from=builder  /usr/local/bin/neptune-lc /usr/local/bin/neptune-lc

CMD ["neptune-lc"]




