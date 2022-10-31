# Build image

This part is the point, different project requirements need to rebuild the image. For a specific introduction to modelbox and the built-in functions of modelbox, please refer to the modelbox documentation manual.

## Container image download

Use the following command to pull the relevant image. For example, cuda11.2, TensorFlow's unbuntu development image, then download the latest version of the image command is as follows:

```shell
docker pull modelbox/modelbox-develop-tensorflow_2.6.0-cuda_11.2-ubuntu-x86_64:latest
```

 The address of the ModelBox image repository is as follows：https://hub.docker.com/u/modelbox 

## One-click startup script

```shell
#!/bin/bash

# ssh map port, [modify]
SSH_MAP_PORT=50022

# editor map port [modify]
EDITOR_MAP_PORT=1104

# http server port [modify]
HTTP_SERVER_PORT=8080

# container name [modify]
CONTAINER_NAME="modelbox_instance_`date +%s` "

# image name
IMAGE_NAME="modelbox/modelbox-develop-tensorflow_2.6.0-cuda_11.2-ubuntu-x86_64"

HTTP_DOCKER_PORT_COMMAND="-p $HTTP_SERVER_PORT:$HTTP_SERVER_PORT"

docker run -itd --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    --tmpfs /tmp --tmpfs /run -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
    --name $CONTAINER_NAME -v /home:/home \
    -p $SSH_MAP_PORT:22 -p $EDITOR_MAP_PORT:1104 $HTTP_DOCKER_PORT_COMMAND \
    $IMAGE_NAME
```

**Notes:**

- After creating a file using the vim start_`docker.sh`, `i` enters the edit mode, pastes the above code, edits and modifications, and saves `wx`.
- In the docker startup script, pay attention to whether the image version launched is consistent with the image version you need.
- If you need to debug `gdb` in the container, you need to add the --privileged parameter to the startup container.
- If you execute the above command on a machine without a `GPU`, you can delete the `--gpus`-related parameters. However, only CPU-related functional units can be used at this time.
- If the port is not occupied but still unreachable after starting mirroring, you need to check the firewall settings.



## Use containers to fulfill requirements

```shell
 docker exec -it [container id] bash
 # Carry out your project.
```

## Build image

1. docker commit image

Save project,It would be more convenient for us to just use `docker commit` directly.

```shell
docker commit [container-ID] [image-name]
```

2. build image

Use the image created

```dockerfile
# load basic image
FROM [image-name]

# configure the Environment variable
ENV PYTHONPATH "/root"

# modify the Working directory
WORKDIR /root

ENTRYPOINT ["python"]
```

Save the content as `Dockerfile`,command:

```shell
docker build -t [image-name] .
```

Get a image of our business needs.