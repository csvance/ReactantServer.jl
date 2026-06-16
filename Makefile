# Container-based image builds for ReactantServer. Targets shell out to a container engine
# (podman by default; rootless is fine). The build context for the images is the repository
# root. The gateway is pure Julia (no Go toolchain or protoc required).
#
#   make            # build the unified node image (default)
#   make image      # build the unified reactantserver node image (supervisor: workers + gateway)
#   make gateway    # build the slim gateway-only image (for multi-node gateway hosts)
#   make worker     # alias for `make image` (kept for the per-GPU-container layout)
#   make e2e        # full-stack end-to-end test (2 GPU workers + gateway; TCP and SHM)
#   make clean      # remove the images this Makefile builds

SHELL := /bin/bash

ENGINE        ?= podman
NODE_IMAGE    ?= reactantserver:latest
GATEWAY_IMAGE ?= reactantserver-gateway:latest
WORKER_IMAGE  ?= reactantserver-worker:latest
LOADGEN_IMAGE ?= reactantserver-loadgen:latest

.PHONY: all image gateway worker loadgen e2e clean help

all: image

## image: build the unified reactantserver node image (large; needs the lib/ submodules checked out)
image:
	$(ENGINE) build -f docker/Dockerfile.worker -t $(NODE_IMAGE) .

## gateway: build the slim pure-Julia gateway-only image
gateway:
	$(ENGINE) build -f docker/Dockerfile.gateway -t $(GATEWAY_IMAGE) .

## worker: alias for `make image`, additionally tagged $(WORKER_IMAGE) for the per-GPU layout
worker: image
	$(ENGINE) tag $(NODE_IMAGE) $(WORKER_IMAGE)

## loadgen: build the dummy-data load generator image (light; no Reactant)
loadgen:
	$(ENGINE) build -f docker/Dockerfile.loadgen -t $(LOADGEN_IMAGE) .

## e2e: full-stack end-to-end test (gateway + two GPU workers via podman; TCP and SHM paths)
e2e:
	bash packages/ReactantServer/test/e2e/run_e2e.sh

## clean: remove the images built by this Makefile (ignores ones that are absent)
clean:
	-$(ENGINE) rmi $(NODE_IMAGE) $(GATEWAY_IMAGE) $(WORKER_IMAGE)

## help: list the available targets
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/^## /  /'
