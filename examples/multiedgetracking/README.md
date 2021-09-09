# Notes about Kafka

- Apache Kafka and Zookeeper are supposed to be deployed in the cloud node.
- In the YAML files, replace the IP/nodename with the one of your cloud node.
- This deployment is a single-pod zookeeper and broker combo.
- Deploying Apache Kafka on the edge node has been tested without success.

# Pre-requisites

- Read carefully the YAML file associated with the computing model. In general, if you have the data in the right place, it should work by default.
- Also, make sure to deploy the pods on the correct nodes (cloud, l2, l1).
- However, there are multiple settings that can be changed and might break the pipeline. This is not supposed to work the first time, so double check the parameters in case something breaks.
- You NEED Kafka if you deploy the `multi-edge-tracking-service-kafka.yaml`. Otherwise it just crashes the pods, no graceful recovery.
- You NEED edgemesh properly working if you deploy `multi-edge-tracking-service.yaml`.

## Cloud node

- Clone the `ai_models` repo in the folder `/data` (yes, in the filesystem root)
- Clone the `yolo_lfs` repo in the folder `/data` (this is required if you want to show the top-10 results because it needs the original images from the gallery)
- Run `cp -r /data/ai_models/toy_containers/merged_bboxes/ /data/ai_models/yolo_lfs/toy_containers/`

## L2 Edge Node

- Clone the `ai_models` repo in the folder `/data`

## L1 Edge Node

- Clone the `yolo_lfs` repo in the folder `/data`
 