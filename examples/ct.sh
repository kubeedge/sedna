#!/bin/bash

prepare(){
action=${1:-create}

kubectl $action -f - <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: ctsecret
  annotations:
    s3-endpoint: 10.44.115.96:9000
    s3-usehttps: "0"
stringData: 
  ACCESS_KEY_ID: minio123
  SECRET_ACCESS_KEY: minio123
EOF

kubectl $action -f - <<EOF
apiVersion: sedna.io/v1alpha1
kind: Dataset
metadata:
  name: "dataset-1"
spec:
  url: "/mnt/data/tt/data/1/robot.txt"
  format: "txt"
  nodeName: euler19
EOF

kubectl $action -f - <<EOF
apiVersion: sedna.io/v1alpha1
kind: Model
metadata:
  name: "yolo-v5-model"
spec:
  url: "/model/yolov5.pth"
  format: "pth"
EOF

kubectl $action -f - <<EOF
apiVersion: sedna.io/v1alpha1
kind: Model
metadata:
  name: "yolo-v5-pretrained-model"
spec:
  url: "/pretrained/yolov5.pth"
  format: "pth"
EOF
}

ctjob(){
action=${1:-create}

kubectl $action -f - <<EOF
apiVersion: sedna.io/v1alpha1
kind: FederatedLearningJob
metadata:
  name: ct-yolo-v5
spec:
  pretrainedModel: # option
    name: "yolo-v5-pretrained-model"
  transmitter: # option
    #ws: { } # option, by default
    s3: # option, but at least one
      aggDataPath: "s3://cotraining"
      credentialName: ctsecret
  aggregationWorker:
    model:
      name: "yolo-v5-model"
    template:
      spec:
        nodeName: "a800-01"
        containers:
          - image: decshub.org/mistnet-yolo-server:v0.4.0
            name: agg-worker
            imagePullPolicy: IfNotPresent
            env: # user defined environments
              - name: "cut_layer"
                value: "4"
              - name: "epsilon"
                value: "100"
              - name: "aggregation_algorithm"
                value: "mistnet"
              - name: "BATCH_SIZE"
                value: "16"
              - name: "EPOCHS"
                value: "100"
            volumeMounts:
                - name: ascend-dirver
                  mountPath: /usr/local/Ascend/driver
                - name: add-ons
                  mountPath: /usr/local/Ascend/add-ons
            resources: # user defined resources
              limits:   
                memory: 32Gi
                huawei.com/Ascend910: 1
            securityContext:
              privileged: true
        volumes:
        - name: ascend-dirver
          hostPath:
              path: /usr/local/Ascend/driver
        - name: add-ons
          hostPath:
               path: /usr/local/Ascend/add-ons
  trainingWorkers:
    - dataset:
        name: "dataset-1"
      template:
        spec:
          nodeName: "euler19"
          containers:
            - image: decshub.org/mistnet-yolo-client:v0.4.0
              name: train-worker
              imagePullPolicy: IfNotPresent
              args: [ "-i", "1" ]
              env: # user defined environments
                - name: "cut_layer"
                  value: "4"
                - name: "epsilon"
                  value: "100"
                - name: "aggregation_algorithm"
                  value: "mistnet"
                - name: "batch_size"
                  value: "32"
                - name: "learning_rate"
                  value: "0.001"
                - name: "epochs"
                  value: "1"
              resources: # user defined resources
                limits:
                  memory: 2.5Gi
                  huawei.com/Ascend310: 1
              securityContext:
                privileged: true
              volumeMounts:
                  - name: ascend-dirver
                    mountPath: /usr/local/Ascend/driver
          volumes:
          - name: ascend-dirver
            hostPath:
                path: /home/data/miniD/driver
EOF
}
