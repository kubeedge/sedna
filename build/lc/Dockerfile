FROM golang:1.14-alpine3.11 AS builder

ARG GO_LDFLAGS

# install build tools
RUN apk update
RUN apk add build-base bash

WORKDIR /code
# copy source
COPY . .
RUN make build WHAT=lc GO_LDFLAGS=$GO_LDFLAGS OUT_DIR=_output


FROM alpine:3.11

COPY --from=builder /code/_output/bin/neptune-lc /usr/local/bin/neptune-lc

CMD ["neptune-lc"]
