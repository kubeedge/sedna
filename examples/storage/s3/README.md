# Using Incremental Learning Job in Helmet Detection Scenario on S3

This example based on the example: [Using Incremental Learning Job in Helmet Detection Scenario](/examples/incremental_learning/helmet_detection/README.md)

### Create a secret with your S3 user credential.

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: mysecret
  annotations:
    s3-endpoint: s3.amazonaws.com # replace with your s3 endpoint e.g minio-service.kubeflow:9000 
    s3-usehttps: "1" # by default 1, if testing with minio you can set to 0
stringData: # use `stringData` for raw credential string or `data` for base64 encoded string
  ACCESS_KEY_ID: XXXX
  SECRET_ACCESS_KEY: XXXXXXXX
```

### Attach the created secret to the Model/Dataset/Job.
`EDGE_NODE` and `CLOUD_NODE` are custom nodes, you can fill it which you actually run.  

```
EDGE_NODE="edge-node" 
CLOUD_NODE="cloud-node"
```

* Attach the created secret to the Model.  

```yaml
kubectl create -f - <<EOF
apiVersion: sedna.io/v1alpha1
kind: Model
metadata:
  name: initial-model
  spec:
    url : "s3://kubeedge/model/base_model"
    format: "ckpt"
    credentialName: mysecret
EOF
```

```yaml
kubectl $action -f - <<EOF
apiVersion: sedna.io/v1alpha1
kind: Model
metadata:
  name: deploy-model
spec:
  url: "s3://kubeedge/model/deploy_model/saved_model.pb"
  format: "pb"
  credentialName: mysecret
EO
```

* Attach the created secret to the Dataset.  

```yaml
kubectl $action -f - <<EOF
apiVersion: sedna.io/v1alpha1
kind: Dataset
metadata:
  name: incremental-dataset
spec:
  url: "s3://kubeedge/data/helmet_detection/train_data/train_data.txt"
  format: "txt"
  nodeName: $CLOUD_NODE
  credentialName: mysecret
EOF

```
* Attach the created secret to the Job(IncrementalLearningJob).

```yaml
kubectl create -f - <<EOF
apiVersion: sedna.io/v1alpha1
kind: IncrementalLearningJob
metadata:
  name: helmet-detection-demo
spec:
  initialModel:
    name: "initial-model"
  dataset:
    name: "incremental-dataset"
    trainProb: 0.8
  trainSpec:
    template:
      spec:
        nodeName: $CLOUD_NODE
        containers:
          - image: kubeedge/sedna-example-incremental-learning-helmet-detection:v0.1.0
            name:  train-worker
            imagePullPolicy: IfNotPresent
            args: ["train.py"]
            env:
              - name: "batch_size"
                value: "32"
              - name: "epochs"
                value: "1"
              - name: "input_shape"
                value: "352,640"
              - name: "class_names"
                value: "person,helmet,helmet-on,helmet-off"
              - name: "nms_threshold"
                value: "0.4"
              - name: "obj_threshold"
                value: "0.3"
    trigger:
      checkPeriodSeconds: 60
      timer:
        start: 02:00
        end: 20:00
      condition:
        operator: ">"
        threshold: 500
        metric: num_of_samples
  evalSpec:
    template:
      spec:
        nodeName: $CLOUD_NODE
        containers:
          - image: kubeedge/sedna-example-incremental-learning-helmet-detection:v0.1.0
            name:  eval-worker
            imagePullPolicy: IfNotPresent
            args: ["eval.py"]
            env:
              - name: "input_shape"
                value: "352,640"
              - name: "class_names"
                value: "person,helmet,helmet-on,helmet-off"
  deploySpec:
    model:
      name: "deploy-model"
    trigger:
      condition:
        operator: ">"
        threshold: 0.1
        metric: precision_delta
    hardExampleMining:
      name: "IBT"
      parameters:
        - key: "threshold_img"
          value: "0.9"
        - key: "threshold_box"
          value: "0.9"
    template:
      spec:
        nodeName: $EDGE_NODE
        containers:
          - image: kubeedge/sedna-example-incremental-learning-helmet-detection:v0.1.0
            name:  infer-worker
            imagePullPolicy: IfNotPresent
            args: ["inference.py"]
            env:
              - name: "input_shape"
                value: "352,640"
              - name: "video_url"
                value: "file://video/video.mp4"
              - name: "HE_SAVED_URL"
                value: "/he_saved_url"
            volumeMounts:
              - name: localvideo
                mountPath: /video/
              - name: hedir
                mountPath: /he_saved_url
            resources:  # user defined resources
              limits:
                memory: 2Gi
        volumes:   # user defined volumes
          - name: localvideo
            hostPath:
              path: /incremental_learning/video/
              type: Directory
          - name: hedir
            hostPath:
              path:  /incremental_learning/he/
              type: Directory
  outputDir: "/incremental_learning/output"
EOF
```
