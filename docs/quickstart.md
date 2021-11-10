
# Quick Start

## Guide
- If you are new to Sedna, you can try the command step by step in this page.
- If you have played the following example, you can find more [examples](/examples/README.md).
- If you want to know more about sedna's architecture and component, you can find them in [sedna home]. 
- If you're looking to contribute documentation improvements, you'll specifically want to see the [kubernetes documentation style guide] before [filing an issue][file-an-issue].
- If you're planning to contribute code changes, you'll want to read the [development preparation guide] next.
- If you're planning to add a new synergy feature directly, you'll want to read the [guide][add-feature-guide] next.

When done, you can also refer our [recommended Git workflow] and [pull request best practices] before submitting a pull request.


[proposals]: /docs/proposals
[development preparation guide]: ./prepare-environment.md
[add-feature-guide]: control-plane/add-a-new-synergy-feature.md

[sedna home]: https://github.com/kubeedge/sedna
[issues]: https://github.com/kubeedge/sedna/issues
[file-an-issue]: https://github.com/kubeedge/sedna/issues/new/choose
[file-a-fr]: https://github.com/kubeedge/sedna/issues/new?labels=kind%2Ffeature&template=enhancement.md

[github]: https://github.com/
[kubernetes documentation style guide]: https://github.com/kubernetes/community/blob/master/contributors/guide/style-guide.md
[recommended Git workflow]: https://github.com/kubernetes/community/blob/master/contributors/guide/github-workflow.md#workflow
[pull request best practices]: https://github.com/kubernetes/community/blob/master/contributors/guide/pull-requests.md#best-practices-for-faster-reviews
[Kubernetes help wanted]: https://www.kubernetes.dev/docs/guide/help-wanted/


The following is showing how to run an incremental learning job by sedna.
## Quick Start 

#### 0. Check the Environment

For Sedna all-in-one installation, it requires you:
  - 1 VM **(one machine is OK, cluster is not required)**
  - 2 CPUs or more
  - 2GB+ free memory, depends on node number setting
  - 10GB+ free disk space
  - Internet connection(docker hub, github etc.)
  - Linux platform, such as ubuntu/centos
  - Docker 17.06+

you can check the docker version by the following command, 
```bash
docker -v
```
after doing that, the output will be like this, that means your version fits the bill.
```
Docker version 19.03.6, build 369ce74a3c
```



#### 1. Deploy Sedna
Sedna provides three deployment methods, which can be selected according to your actual situation:

- [Install Sedna AllinOne](setup/all-in-one.md). (used for development, here we use it)
- [Install Sedna local up](setup/local-up.md).  
- [Install Sedna on a cluster](setup/install.md).

The [all-in-one script](/scripts/installation/all-in-one.sh) is used to install Sedna along with a mini Kubernetes environment locally, including:
  - A Kubernetes v1.21 cluster with multi worker nodes, default zero worker node.
  - KubeEdge with multi edge nodes, default is latest KubeEdge and one edge node.
  - Sedna, default is the latest version.

  ```bash
  curl https://raw.githubusercontent.com/kubeedge/sedna/master/scripts/installation/all-in-one.sh | NUM_EDGE_NODES=2 bash -
  ```

#### 2. Download model and datasets
In incremental_learning jobs, the following files will be indispensable:

- base model: tensorflow object detection Fine-tuning a model from an existing checkpoint.
- deploy model:  tensorflow object detection model, for inference.
- train data: images with label use for Fine-tuning model.
- test data: video stream use for model inference.

```bash
# download models, including base model and deploy model.

cd /
wget https://kubeedge.obs.cn-north-1.myhuaweicloud.com/examples/helmet-detection/models.tar.gz
tar -zxvf models.tar.gz

# download train data

cd /data/helmet_detection  # notes: files here will be monitored and used to trigger the incremental training
wget  https://kubeedge.obs.cn-north-1.myhuaweicloud.com/examples/helmet-detection/dataset.tar.gz
tar -zxvf dataset.tar.gz

# download test data

cd /incremental_learning/video/
wget https://kubeedge.obs.cn-north-1.myhuaweicloud.com/examples/helmet-detection/video.tar.gz
tar -zxvf video.tar.gz
 
```

#### 3. Create model and dataset object

In this example, `$WORKER_NODE` is a custom node, you can fill it which you actually run.  

```
WORKER_NODE="edge-node" 
```

- Create Dataset  

```
kubectl create -f - <<EOF
apiVersion: sedna.io/v1alpha1
kind: Dataset
metadata:
  name: incremental-dataset
spec:
  url: "/data/helmet_detection/train_data/train_data.txt"
  format: "txt"
  nodeName: $WORKER_NODE
EOF
```

- Create Initial Model to simulate the initial model in incremental learning scenario.  

```
kubectl create -f - <<EOF
apiVersion: sedna.io/v1alpha1
kind: Model
metadata:
  name: initial-model
spec:
  url : "/models/base_model"
  format: "ckpt"
EOF
```

- Create Deploy Model  

```
kubectl create -f - <<EOF
apiVersion: sedna.io/v1alpha1
kind: Model
metadata:
  name: deploy-model
spec:
  url : "/models/deploy_model/saved_model.pb"
  format: "pb"
EOF
```

#### 4. Start an incremental learning job

* create the job:

```
kubectl -f https://raw.githubusercontent.com/kubeedge/sedna/main/examples/incremental_learning/helmet_detection/helmet_detection.yaml
```

1. The `Dataset` describes data with labels and `HE_SAVED_URL` indicates the address of the deploy container for uploading hard examples. Users will mark label for the hard examples in the address.
2. Ensure that the path of outputDir in the YAML file exists on your node. This path will be directly mounted to the container.


#### 5. Check the result

Query the service status:  

```
kubectl get incrementallearningjob helmet-detection-demo
```

In the `IncrementalLearningJob` resource helmet-detection-demo, the following trigger is configured:  

```
trigger:
  checkPeriodSeconds: 60
  timer:
    start: 02:00
    end: 20:00
  condition:
    operator: ">"
    threshold: 500
    metric: num_of_samples
```

## API

- control-plane: Please refer to this [link](api/crd).
- Lib: Please refer to this [link](api/lib).

## Contributing

Contributions are very welcome!

- control-plane: Please refer to this [link](contributing/control-plane/development.md).
- Lib: Please refer to this [link](contributing/lib/development.md).

## Community

Sedna is an open source project and in the spirit of openness and freedom, we welcome new contributors to join us. 
You can get in touch with the community according to the ways:
* [Github Issues](https://github.com/kubeedge/sedna/issues)
* [Regular Community Meeting](https://zoom.us/j/4167237304)
* [slack channel](https://app.slack.com/client/TDZ5TGXQW/C01EG84REVB/details)

