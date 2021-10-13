This document is intended to provide contributors with an introduction to develop [LC(LocalController)][framework] of the Sedna project. 

### Debug LC
1\. config LC:

Setting up the environments:
1. `GM_ADDRESS`: the addresss of GM.
1. `NODENAME`: the node name at which LC running.
1. `ROOTFS_MOUNT_DIR`: the directory of the host mounts, default `/rootfs`.

```shell
# update these values if neccessary
export GM_ADDRESS=192.168.0.10:9000
export NODENAME=edge-node
export ROOTFS_MOUNT_DIR=""
```

> **Note**: If you have already run Sedna by following the [install doc], and decide to run LC in-place, you don't need to setup these environments, run `make lcimage` and `kubectl -n sedna delete pod lc-<pod-name>`.

2\. compile and run LC directly:

```shell
make WHAT=lc
_output/bin/sedna-lc -v4
```

Alternatively you can debug LC with [golang delve]:

```shell
dlv debug cmd/sedna-lc/sedna-lc.go -- -v4
```

[install doc]: /docs/setup/install.md
[golang delve]: https://github.com/go-delve/delve
[framework]: /docs/proposals/architecture.md#architecture
