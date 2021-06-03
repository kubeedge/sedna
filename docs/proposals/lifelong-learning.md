* [Lifelong Learning](#lifelong-learning)
   * [Motivation](#motivation)
     * [Goals](#goals)
     * [Non\-goals](#non-goals)
   * [Proposal](#proposal)
     * [Use Cases](#use-cases)
   * [Design Details](#design-details)
     * [CRD API Group and Version](#crd-api-group-and-version)
     * [Lifelong learning CRD](#lifelong-learning-crd)
     * [Lifelong learning type definition](#lifelong-learning-job-type-definition)
     * [Lifelong learning sample](#lifelong-learning-job-sample)
     * [Validation](#validation)
   * [Controller Design](#controller-design)
     * [Lifelong Learning Controller](#lifelong-learning-controller)
     * [Downstream Controller](#downstream-controller)
     * [Upstream Controller](#upstream-controller)
     * [Details of api between GM(cloud) and LC(edge)](#details-of-api-between-gmcloud-and-lcedge)
   * [Workers Communication](#workers-communication)

# Lifelong Learning
## Motivation


At present, edge-cloud synergy machine learning is confronted with the challenge of heterogeneous data distributions in complex scenarios and small samples on the edge. The edge-cloud synergy lifelong learning is accordingly proposed: 1) In order to learn with shared knowledge between historical scenarios, the scheme is essentially the combination of another two learning schemes, i.e., multi-task learning and incremental learning; 2) The cloud knowledge base in lifelong learning empowers the scheme with memory ability, which helps to adapt historical knowledge to new and unseen situations on the edge. Joining the forces of multi-task learning, incremental learning and the knowledge base, the lifelong learning scheme seeks to fundamentally overcome the above challenges of edge-cloud synergy machine learning.
### Goals


* edge-cloud collaborative continuous learning.
* Knowledge sharing across the edge of the cloud.
* Automatic discovery and transfer learning of new knowledge.

## Proposal
We propose using Kubernetes Custom Resource Definitions (CRDs) to describe 
the lifelong learning specification/status and a controller to synchronize these updates between edge and cloud.

![](./images/lifelong-learning-job-crd.png)

### Use Cases

* Users can create the lifelong learning jobs, by providing training scripts, configuring training hyperparameters, providing training datasets, configuring training and deployment triggers.



## Design Details
There are three stages in a lifelong learning job: train/eval/deploy.

Each stage contains these below states:
1. Waiting: wait to trigger satisfied, i.e. wait to train/eval/deploy
1. Ready: the corresponding trigger satisfied, now ready to train/eval/deploy
1. Starting: the corresponding stage is starting
1. Running: the corresponding stage is running
1. Failed: the corresponding stage failed
1. Completed: the corresponding stage completed

### CRD API Group and Version
The `LifelongLearningJob` CRD will be namespace-scoped.
The tables below summarize the group, kind and API version details for the CRD.

* LifelongLearningJob

| Field                 | Description             |
|-----------------------|-------------------------|
|Group                  | sedna.io     |
|APIVersion             | v1alpha1                |
|Kind                   | LifelongLearningJob             |

### Lifelong learning CRD
See the [crd source](/build/crds/sedna/Lifelonglearningjob_v1alpha1.yaml) for details.

### Lifelong learning job type definition

See the [golang source](/pkg/apis/sedna/v1alpha1/Lifelongllearningjob_types.go) for details.

#### Validation
[Open API v3 Schema based validation](https://kubernetes.io/docs/tasks/access-kubernetes-api/custom-resources/custom-resource-definitions/#validation) can be used to guard against bad requests.
Invalid values for fields (example string value for a boolean field etc) can be validated using this.

Here is a list of validations we need to support :
1. The `dataset` specified in the crd should exist in k8s.
1. The edgenode name specified in the crd should exist in k8s.

### Lifelong learning job sample
See the [source](/build/crd-samples/sedna/Lifelonglearningjob_v1alpha1.yaml) for an example.
    
## Controller Design

The Lifelong learning controller starts three separate goroutines called `upstream`, `downstream` and `Lifelonglearningjob`controller.<br/>
These are not separate controllers as such but named here for clarity.
- Lifelong learning: watch the updates of lifelong-learning job crds, and create the workers depending on the state machine.
- downstream: synchronize the lifelong-learning-job updates from the cloud to the edge node.
- upstream: synchronize the lifelong-learning-job updates from the edge to the cloud node.

### Lifelong Learning Controller
![](./images/lifelong-learning-controller.png)

The lifelong-learning controller watches for the updates of lifelong-learning jobs and the corresponding pods against the K8S API server.<br/>
Updates are categorized below along with the possible actions:

| Update Type                    | Action                                       |
|-------------------------------|---------------------------------------------- |
|New lifelong-learning-job Created             | Wait to train trigger satisfied|
|lifelong-learning-job Deleted                 | NA. These workers will be deleted by [k8s gc](https://kubernetes.io/docs/concepts/workloads/controllers/garbage-collection/).|
|The Status of lifelong-learning-job Updated               | Create the train/eval worker if it's ready.|
|The corresponding pod created/running/completed/failed                 | Update the status of lifelong-learning job.|

### Downstream Controller
![](./images/lifelong-learning-downstream-controller.png)

The downstream controller watches for the lifelong-learning job updates against the K8S API server.<br/>
Updates are categorized below along with the possible actions that the downstream controller can take:

| Update Type                    | Action                                       |
|-------------------------------|---------------------------------------------- |
|New Lifelong-learning-job Created             |Sends the job information to LCs.|
|Lifelong-learning-job Deleted                 | The controller sends the delete event to LCs.|

### Upstream Controller
![](./images/lifelong-learning-upstream-controller.png)

The upstream controller watches for the lifelong-learning job updates from the edge node and applies these updates against the API server in the cloud.<br/>
Updates are categorized below along with the possible actions that the upstream controller can take:

| Update Type                        | Action                                        |
|-------------------------------     |---------------------------------------------- |
|Lifelong-learning-job Reported State Updated    |  The controller appends the reported status of the job by LC in the cloud. |

### Details of api between GM(cloud) and LC(edge)
1. GM(downstream controller) syncs the job info to LC:
    ```go
    // POST <namespace>/lifelonglearningjobs/<job-name>
    // body same to the job crd of k8s api, omitted here.
    ```

1. LC uploads the job status which reported by the worker to GM(upstream controller):
    ```go
    // POST <namespace>/lifelonglearningjobs/<job-name>/status
   
    // WorkerMessage defines the message from that the training worker. It will send to GM.
    type WorkerMessage struct {
        Phase  string        `json:"phase"`
        Status string        `json:"status"`
        Output *WorkerOutput `json:"output"`
    }
    // 
    type WorkerOutput struct {
        Models   []*Model  `json:"models"`
        OwnerInfo *OwnerInfo `json:"ownerInfo"`
    }
    
    // Model defines the model information 
    type Model struct {
        Format  string             `json:"format"`
        URL     string             `json:"url"`
        // Including the metrics, e.g. precision/recall
        Metrics map[string]float64 `json:"metrics"`
    }
    
    // TaskInfo defines the task information
    type TaskInfo struct {
        // Current training round
        CurrentRound int    `json:"currentRound"`
        UpdateTime   string `json:"updateTime"`
    }
    ```    

### The flows of lifelong learning job
- Flow of the job creation:

![](./images/lifelong-learning-flow-creation.png)

- Flow of the `train` stage:

![](./images/lifelong-learning-flow-train-stage.png)

- Flow of the `eval` stage:

![](./images/lifelong-learning-flow-eval-stage.png)

- Flow of the `deploy` stage:

![](./images/lifelong-learning-flow-deploy-stage.png)

## Workers Communication
No need to communicate between workers.
