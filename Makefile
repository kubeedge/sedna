# Copyright 2021 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

GOPATH ?= $(shell go env GOPATH)

OUT_DIR ?= _output
OUT_BINPATH := $(OUT_DIR)/bin
OUT_IMAGESPATH := $(OUT_DIR)/images

IMAGE_REPO ?= kubeedge

# the env PLATFORMS defines to generate linux images for amd 64-bit, arm 64-bit and armv7 architectures
# the full list of PLATFORMS is linux/amd64,linux/arm64,linux/arm/v7
PLATFORMS ?= linux/amd64,linux/arm64
COMPONENTS ?= gm lc kb

IMAGE_TAG ?= v0.3.0
GO_LDFLAGS ?= ""

# set allowDangerousTypes for allowing float
CRD_OPTIONS ?= "crd:crdVersions=v1,allowDangerousTypes=true"

# make all builds both gm and lc binaries
BINARIES=gm lc
SHELL=/bin/bash

.EXPORT_ALL_VARIABLES:

define BUILD_HELP_INFO
# Build code with verifying or not.
# target all is the "build" with verify.
# Args:
#   WHAT: binary names to build. support: $(BINARIES)
#         the build will produce executable files under ./$(OUT_BINPATH)
#         If not specified, "everything" will be built.
#
# Example:
#   make TARGET
#   make TARGET HELP=y
#   make TARGET WHAT=gm
#   make TARGET WHAT=lc GOLDFLAGS="" GOGCFLAGS="-N -l"
#     Note: Specify GOLDFLAGS as an empty string for building unstripped binaries, specify GOGCFLAGS
#     to "-N -l" to disable optimizations and inlining, this will be helpful when you want to
#     use the debugging tools like delve. When GOLDFLAGS is unspecified, it defaults to "-s -w" which strips
#     debug information, see https://golang.org/cmd/link for other flags.

endef

.PHONY: build docker-cross-build all
ifeq ($(HELP),y)
build all:
	@echo "$${BUILD_HELP_INFO//TARGET/$@}"
else
# build without verify
# default target
build:
	hack/make-rules/build.sh $(WHAT)
# build multi-platform images and results will be saved in tar packages.
docker-cross-build:
	bash hack/make-rules/cross-build.sh

all: verify build

endif


define VERIFY_HELP_INFO
# verify golang, vendor, vendor licenses and codegen
#
# Example:
# make verify
endef
.PHONY: verify
ifeq ($(HELP),y)
verify:
	@echo "$$VERIFY_HELP_INFO"
else
verify: verify-golang verify-vendor verify-codegen verify-vendor-licenses
endif

.PHONY: verify-golang
verify-golang:
	hack/verify-golang.sh

.PHONY: verify-vendor
verify-vendor:
	hack/verify-vendor.sh
.PHONY: verify-codegen
verify-codegen:
	hack/verify-codegen.sh
.PHONY: verify-vendor-licenses
verify-vendor-licenses:
	hack/verify-vendor-licenses.sh

define LINT_HELP_INFO
# run golang lint check.
#
# Example:
#   make lint
#   make lint HELP=y
endef
.PHONY: lint
ifeq ($(HELP),y)
lint:
	@echo "$$LINT_HELP_INFO"
else
lint:
	hack/make-rules/lint.sh
endif

define PYLINT_HELP_INFO
# run python lint check.
#
# Example:
#   make pylint
#   make pylint HELP=y
endef
.PHONY: pylint
ifeq ($(HELP),y)
pylint:
	@echo "$$PYLINT_HELP_INFO"
else
pylint:
	hack/make-rules/pylint.sh
endif

define CLEAN_HELP_INFO
# Clean up the output of make.
#
# Example:
#   make clean
#   make clean HELP=y
#
endef
.PHONY: clean
ifeq ($(HELP),y)
clean:
	@echo "$$CLEAN_HELP_INFO"
else
clean:
	hack/make-rules/clean.sh
endif

.PHONY: images gmimage lcimage kbimage
images: gmimage lcimage kbimage
gmimage lcimage kbimage:
	docker build --build-arg GO_LDFLAGS=${GO_LDFLAGS} -t ${IMAGE_REPO}/sedna-${@:image=}:${IMAGE_TAG} -f build/${@:image=}/Dockerfile .


.PHONY: push push-examples push-all push-multi-platform-images
push-all: push-multi-platform-images push-examples

# push target pushes sedna-built images
push: images
	for target in $(COMPONENTS); do \
  		docker push ${IMAGE_REPO}/sedna-$$target:${IMAGE_TAG}
	done
	bash scripts/storage-initializer/push_image.sh

push-examples:
	bash examples/push_image.sh

# push multi-platform images
push-multi-platform-images:
	bash hack/make-rules/push.sh

.PHONY: e2e
e2e:
	hack/run-e2e.sh

# Generate CRDs by kubebuilder
.PHONY: crds controller-gen
crds: controller-gen
	$(CONTROLLER_GEN) $(CRD_OPTIONS) paths="./pkg/apis/sedna/v1alpha1" output:crd:artifacts:config=build/crds

# Get the currently used golang install path (in GOPATH/bin, unless GOBIN is set)
ifeq (,$(shell go env GOBIN))
GOBIN=$(shell go env GOPATH)/bin
else
GOBIN=$(shell go env GOBIN)
endif

# find or download controller-gen
# download controller-gen if necessary
controller-gen:
ifeq (, $(shell which controller-gen))
	@{ \
	set -e ;\
	CONTROLLER_GEN_TMP_DIR=$$(mktemp -d) ;\
	cd $$CONTROLLER_GEN_TMP_DIR ;\
	go mod init tmp ;\
	go get sigs.k8s.io/controller-tools/cmd/controller-gen@v0.4.1 ;\
	rm -rf $$CONTROLLER_GEN_TMP_DIR ;\
	}
CONTROLLER_GEN=$(GOBIN)/controller-gen
else
CONTROLLER_GEN=$(shell which controller-gen)
endif
