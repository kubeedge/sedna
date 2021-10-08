# List of Changes

Below are listed the changes applied to the original Sedna repository divided into groups. The changes are flagged with the following tags:

- [I] Invasive: Affects the Sedna framework as a whole and can potentially introduce breaking changes. Requires recompilation.
- [SC] Self-contained: Can be integrated without affecting other modules in Sedna (e.g., Kafka module).
- [ND] Effect on Sedna is non-determined.
- [S/M/B] Size of the features: Small/Medium/Big

## Applications

This changes can be found in the `examples` folder:

- [I][B] Added `multiedgetracking` example.
- [I][B] Added `dnn_partinioning` example.

## Sedna Library

This changes can be found in the `sedna\lib` folder:

- [SC][S] Added optical flow analysis algorithm (LukasKanade) to `algorithms` folder.
- [I][B] Added `torch` folder under `backend` to support usage of PyTorch models.
- [SC][S] Added `banchmark.py` under `common` to collect metrics about function execution time in Sedna the capability to interact with Fluentd.
- [I][S] Modified `log.py` to support change of log level from the container dockerfile and to printout JSON formatted logs (necessary for monitoring and parsing with Fluentd).
- [SC][M] Added under `core` the folders for the new examples: `dnn_partitioning` and `multi_edge_tracking`.
- [I][S] Modified `base.py` to move from base class application specific properties.
- [SC][M] Added `datasources/kafka` folder to add Apache Kafka support for Sedna services (producer/consumer).
- [SC][M] Added new services in `service` folder.

## Controller (GM)

This changes can be found in the `pkg` folder:

- [I][B] Added `multiedgetrackingservice_types.go` and `dnnpartitioningservice_types.go` to `v1alpha1` to support the newly created example applications.
- [I][B] Added `multiedgetrackingservice.go` and `dnnpartitioningservice.go` to `globalmanager` to support the newly created example applications.
- [I][M] Added new functions in `common.go` and `worker.go` which are used by the new controllers.

## Extra

- [I][M] Modified the `build_image.sh` file to building Docker images by example rather than building all of them.
- [SC][S] Modified the Sedns requirements.txt to load the extra `kafka-python` and `fluent-logger` dependecies.

