* [Observability Management](#observability-management)
  * [Motivation](#motivation)
    * [Goals](#goals)
  * [Proposal](#proposal)
  * [Design Details](#design-details)
    * [Monitoring Metrics](#monitoring-metrics)
    * [Collecting Logs](#collecting-logs)
    * [Display](#display)
  * [Key Deliverable](#key-deliverable)
  * [Road Map](#roadmap)

# Observability Management

## Motivation
Currently, users can only check the status, parameters and metrics of tasks via the command line after creating edge-cloud synergy AI tasks by sedna.

This proposal provides observability management for displaying logs and metrics to monitor tasks in real time, so that users can easily check the status, parameters and metrics of tasks.

### Goals
* The metrics and status of Sedna's components, such as Global Manager, Local Controllers and Workers, can be monitored.
* The parameters of edge-cloud synergy AI tasks like the count of inference can be collected and displayed on the Observability management.
* Logs of all pods created by Sedna can be collected to manage and display.
* Observability data collected can be displayed on Grafana appropriately and aesthetically.

## Proposal
We propose using Prometheus and Loki to collect observability data like metrics and logs.
And the Observability data can be displayed with functions of Grafana.

## Design Details
![](./images/observability-management-architecture.png)

### Monitoring Metrics
Sedna consists of GlobalManager, LocalControllers and Workers, which ensure Sedna works.
The observability management can monitor these components from different aspects.

#### Common System Metrics
The running status of cluster resource objects like pods, service, deployment on edge and cloud can be monitored by kube-state-metrics.

| **Metric**           | **Description**                                       |
|----------------------|-------------------------------------------------------|
| PodStatus            | Running / Pending / Failed                            | 
| ContainerStatus      | Running / Waiting / Terminated                        | 
| K8sJobStatus         | Succeeded / Failed                                    |

Especially for sedna, the running status of CRDs like JointInferenceService, IncrementalLearningJob and FederatedLearningJob can also be monitored.

| **Metric**           | **Description**                                           |
|----------------------|-----------------------------------------------------------|
| StageConditionStatus | Waiting / Ready / Starting / Running / Completed / Failed |  
| StartTime            | The start time of tasks                                   |  
| CompletionTime       | The completion time of tasks                              |
| SampleNum            | The number of samples                                     |

#### Algorithm Metrics (Designed Individually For Each Task)
For edge-cloud synergy AI tasks, we create customized exporters to collect the metrics we need in different types of tasks.

##### Joint Inference
| **Metric**           | **Description**                                           |
|----------------------|-----------------------------------------------------------|
| EdgeInferenceCount   | The count of inference at edge                            |
| CloudInferenceCount  | The count of inference at cloud                           |
| HardSampleNum        | The number of hard samples                                |


##### Incremental Learning

| **Metric**                | **Description**                                                                               |
|---------------------------|-----------------------------------------------------------------------------------------------|
| LearningRate              | The learning rate at training stage                                                           |
| EpochNum                  | The number of epochs at training stage                                                        |
| BatchSize                 | The batch size at training stage                                                              |
| TrainSampleNum            | The number of samples used at training stage                                                  |
| EvalSampleNum             | The number of samples used at eval stage                                                      |
| TaskStage                 | Training Stage / Eval Stage / Deploying Stage                                                 | 
| CurrentStatus             | The status of current stage like running and waiting (only for training stage and eval stage) |
| IterationNo               | The current iteration No. of incremental learning                                             |
| IterationInferenceCount   | The count of inference at current iteration                                                   |
| IterationHardSampleNum    | The number of hard samples at current iteration                                               |
| EvalNewModelPath          | The path of new model at eval stage                                                           |
| AccuracyMetricForNewModel | mAP / Precision / Recall / F1-score for new model                                             |
| AccuracyMetricForOldModel | mAP / Precision / Recall / F1-score for old model                                             |
| TrainOutputModelPath      | The output path for model after training                                                      |
| DeployModelPath           | The path for deploying model                                                                  |
| DeployStatus              | Waiting / OK / Not OK                                                                         |
| TrainRemainingTime        | The remaining time at train stage                                                             |
| TrainLoss                 | loss at train stage                                                                           |
| EvalRemainingTime         | The remaining time at eval stage                                                              |

##### Federated Learning

| **Metric**              | **Description**                                           |
|-------------------------|-----------------------------------------------------------|
| TrainNodes              | The nodes participating in training                       |
| TrainStatus             | Current training status                                   |
| NodeSampleNum           | The number of samples at each node                        |
| IterationNo             | The current iteration No. of federated learning           |
| AggregationNo           | The current aggregation No. of federated learning         |

### Collecting Logs
We plan to use Loki to collect logs from all pods created by Sedna like Local Controllers, Global Manager and workers derived from tasks.

### Display

Grafana supports presentation of customised query based on Prometheus and Loki monitored data via PromQL and LogQL.
For metrics, data can be displayed as line graphs, pie charts, histogram and so on.
For logs, querying can be performed to show all logs matched and the number of matched logs at different times.

## Key Deliverable
* Open-Source software deployment and usage guide
* Code and configuration files
  * Code of edge-cloud synergy AI task exporters
  * Component configuration files for Prometheus, Loki and Grafana
  * Exporter configuration files for node-exporter and kube-state-metrics
  * JSON configuration files for easy-to-use and good-looking panels on Grafana
* End-to-end test cases

## Roadmap
* July 2022:
  * Complete Prometheus configuration files writing for node monitoring and cluster resource objects monitoring.
  * Finish Loki configuration files writing for logs management.
* August 2022:
  * Create the customized exporters to collect metrics of tasks.
  * Design easy-to-use and good-looking panels and display the observability data on Grafana.
* September 2022:
  * Design test cases.
  * Write the document for deploying observability management.
