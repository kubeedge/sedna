# Development Guide
This document is intended to provide contributors with an introduction to developing the control plane of the Sedna project. 
There are two components in our control plane: [GM(Global Manager) and LC(Local Controller)][framework].

## Overview

Sedna provides various utilities for development wrapped in `make`.

Most scripts require more or less only `make` + `bash` on the host, and generally
stick to POSIX utilities. Some scripts use `docker` e.g. for image building or
to use docker images containing special tools.

## Read Conventions Before Coding
- [k8s coding convention]
- [k8s api convention] if you want to add or update api.

## Building
Before continuing, you need to follow the [prerequisites installation guide] if you haven't done it yet.

### Building Code

```shell
# build GM
make WHAT=gm

# build LC
make WHAT=lc

# build GM/LC both
make
```

`_output/bin` will contain the freshly built binaries `sedna-gm` and `sedna-lc` upon a successful build.

### Add or Update API
If you add or update APIs in `pkg/apis/`, you need to run:
1. run `bash hack/update-codegen.sh` to update client-go code in `pkg/client`.
	> **Note**: you need to checkout the code to `$GOPATH/src/github.com/kubeedge/sedna`.
1. run `make crds` to update the api definition.
1. run `kubectl replace -f build/crds/` to update your kubernetes environment.


### Add or Update Dependencies

Run the following commands to update [Go Modules]:

```
go mod tidy
go mod vendor

# Or: hack/update-vendor.sh
```

Run `hack/update-vendor-licenses.sh` to update [vendor licenses](/LICENSES).

### Running Sedna

- [See here to run GM](debug-gm.md)
- [See here to run LC](debug-lc.md)


### Building Images

To build the GM/LC base images:

```shell
# build GM
make gmimage

# build LC
make lcimage
```

## Run verification
You need to run the verification tests locally to give your pull request the best chance of being accepted.

To run all verification tests, use this command:

```shell
make verify
```

## Debugging
TBD


## Testing
TBD

### E2E Testing
TBD

## Linting
You can run all of our lints at once with `make lint`.

Lints include:
- [golangci-lint] with [a custom config](/.golangci.yml) to lint Go sources.


## CI
We use [GitHub Actions] which are configured in [.github/workflows](/.github/workflows) in the Sedna repo.

[golangci-lint]: https://github.com/golangci/golangci-lint
[GitHub Actions]: https://github.com/features/actions
[go modules]: https://github.com/golang/go/wiki/Modules
[k8s coding convention]: https://github.com/kubernetes/community/blob/master/contributors/guide/coding-conventions.md
[k8s api convention]: https://github.com/kubernetes/community/blob/master/contributors/devel/sig-architecture/api-conventions.md
[prerequisites installation guide]: /docs/contributing/prepare-environment.md
[framework]: /docs/proposals/architecture.md#architecture
