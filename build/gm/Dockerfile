FROM golang:1.14-alpine3.11 AS builder

ARG GO_LDFLAGS

WORKDIR /code

# copy source
COPY . .

RUN CGO_ENABLED=0 go build -o /usr/local/bin/neptune-gm -ldflags "$GO_LDFLAGS -w -s" \
cmd/neptune-gm/neptune-gm.go


FROM alpine:3.11

COPY --from=builder /usr/local/bin/neptune-gm /usr/local/bin/neptune-gm

COPY build/gm/gm-config.yaml /gm.yaml

CMD ["neptune-gm", "--config", "/gm.yaml"]
