GOPATH ?= $(shell go env GOPATH)

# make all builds both gm and lc binaries
BINARIES=gm lc
SHELL=/bin/bash

.EXPORT_ALL_VARIABLES:
OUT_DIR ?= _output
OUT_BINPATH := $(OUT_DIR)/bin

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

.PHONY: build all
ifeq ($(HELP),y)
build all:
	@echo "$${BUILD_HELP_INFO//TARGET/$@}"
else
# build without verify
# default target
build:
	hack/make-rules/build.sh $(WHAT)

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


IMAGE_REPO ?= ghcr.io/kubeedge/sedna
IMAGE_TAG ?= v1alpha1
GO_LDFLAGS ?=''

.PHONY: images gmimage lcimage
images: gmimage lcimage
gmimage lcimage:
	docker build --build-arg GO_LDFLAGS=${GO_LDFLAGS} -t ${IMAGE_REPO}/${@:image=}:${IMAGE_TAG} -f build/${@:image=}/Dockerfile .
