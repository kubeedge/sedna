* [LLM Joint Inference](#llm-joint-inference)
* [Motivation](#motivation)
* [Goals](#goals)
* [Proposal](#proposal)
  * [Technology selection](#technology-selection)
* [Design Details](#design-details)
  * [Overall Architecture](#overall-architecture)
  * [two mode](#two-mode)
  * [bacth deployment](#bacth-deployment)
  * [implementation](#implementation)
    * [API Design](#api-design)
    * [example](#example)
  * [Development Progress Planning](#development-progress-planning)
    * [Phase 1](#phase-1)
    * [Phase 2](#phase-2)
    * [Phase 3](#phase-3)
    * [Phase 4](#phase-4)

# LLM Joint Inference
## Motivation

Sedna leverages the edge-cloud collaboration capabilities of the edge computing framework KubeEdge to achieve cross-edge-cloud collaborative training and inference. Joint inference is an important application of Sedna's edge-cloud collaborative inference, which effectively improves the efficiency and accuracy of edge-side inference. However, there are still some deficiencies in the ease of use and scalability of the current Sedna joint inference, mainly manifested in the following aspects:

1. Currently, the application development of joint inference strongly depends on the SDK toolkit provided by Sedna. For users, getting familiar with the Sedna SDK is a matter with a high learning cost. Moreover, it also involves the compilation of Docker images. For some pure algorithm engineers, they may not be familiar with the theory of cloud-native, which undoubtedly increases the difficulty of getting started.
2. The current design of Sedna's joint inference can only meet the joint inference tasks of traditional discriminative models, and it is difficult to adapt to the existing large language model algorithms.
3. Sedna's joint inference can only achieve one-to-one deployment, and cannot achieve one-to-many deployment.
4. Sedna's joint inference forces the deployment of edge-cloud inference models and does not support the single-cloud or single-edge mode.

Based on the above problems, this proposal aims to introduce the following functional enhancement solutions:

1. Decouple the SDK from the Sedna framework. Users only need to perform simple model configuration to start the model service, without the need for instance development based on the SDK.
2. Support the configuration of both discriminative models and large language models at the same time.
3. Support the many-to-one configuration of joint inference tasks (that is, batch-deploy multiple edge-side models and share the same cloud-side model).
4. Do not force users to use the joint inference mode, and support the single-edge or single-cloud mode.


# Goals

In summary, the goal of this reconstruction of the joint inference paradigm is:

1. Implement a new inference paradigm in Sedna to achieve the above-mentioned functions (Since there are already significant differences in functionality compared to the original inference paradigm, we will conduct development on the new version v1alpha2.).
2. Provide two examples corresponding to the traditional discriminative model and the large language model respectively.
3. Add necessary test cases.
4. Update the Sedna documentation to reflect the new inference paradigm.

# Proposal

## Technology selection

Currently, the mainstream inference frameworks mainly include Triton, vLLM, SGLang, lmdeploy and Ollama. The following is a detailed comparison of them.

|                                       | [Triton](https://github.com/triton-inference-server/server) | [vLLM](https://github.com/vllm-project/vllm)        | [SGLang](https://github.com/sgl-project/sglang) | [LMDeploy](https://github.com/InternLM/lmdeploy)     | [Ollama](https://github.com/ollama/ollama) |
|---------------------------------------|-------------------------------------------------------------|-----------------------------------------------------|-------------------------------------------------|------------------------------------------------------|--------------------------------------------|
| **Core Positioning**                  | Industrial-grade multi-model inference service              | High-throughput inference for large language models | Structured generation control engine            | Enterprise-level edge-cloud collaborative deployment | Local lightweight deployment tool          |
| **Single GPU Throughput** (Llama3-8B) | 1800 tokens/s                                               | **4200 tokens/s**                                   | 3200 tokens/s                                   | 3800 tokens/s                                        | 850 tokens/s                               |
| **Memory Utilization Rate**           | 65%-75%                                                     | **92%-95%**                                         | 80%-85%                                         | 88%-90%                                              | 50%-60%                                    |
| **First Token Time**                  | 18ms                                                        | **9ms**                                             | 12ms                                            | 11ms                                                 | 35ms                                       |
| **Model Support**                     | PyTorch/TensorFlow/ONNX                                     | Llama3/GPT-5/Mistral                                | Structured generation in JSON/YAML format       | HuggingFace/AWQ/TensorRT                             | GGML/GGUF/AWQ                              |
| **Core Technological Innovation**     | Orchestration of cross-model pipelines                      | Dynamic KV cache sharding                           | Fast parsing of xGrammar                        | Dynamic Batching & GPU Memory Compression            | Hot replacement of local models            |
| **Typical Latency** (batch=32)        | 22ms/token                                                  | 15ms/token                                          | 18ms/token                                      | **13ms/token**                                       | 45ms/token                                 |
| **Hardware Compatibility**            | The whole series of NVIDIA products/AMD MI300               | NVIDIA/AMD ROCm                                     | NVIDIA/Intel GPU                                | NVIDIA / Acend 910B                                  | CPU/Apple Silicon                          |
| **Open Source License**               | BSD-3                                                       | Apache 2.0                                          | MIT                                             | Apache 2.0                                           | AGPL                                       |
| **Applicable Field**                  | Industrial-grade AI services, CV/NLP multimodality          | Commercial LLM applications, AI assistant backends  | Code generation, dialogue systems               | Edge-cloud collaborative deployment                  | Personal research, prototype development   |

Through the above comparison, it can be found that vLLM, LMDeploy and SGLang can only provide services for large language models (LLMs) and cannot provide services for discriminative models (such as CV models). On the other hand, Ollama is designed for local experiments and is not suitable for large-scale production environments.
Triton as a general-purpose framework, although it has some performance gaps compared with vLLM, LMDeploy and SGLang, it performs relatively balanced in all aspects and can support both traditional discriminative models and large language models(LLMs) simultaneously. Moreover, it can also support both GPU and CPU inference, making it quite suitable for our application scenarios.

However, Triton also has its drawbacks: 
- Firstly, as mentioned before, there is a certain performance gap compared with vLLM and SGLang. 
- Secondly, the inference tasks of Triton on GPUs strongly rely on Nvidia and it cannot support GPUs from manufacturers such as Ascend and Cambricon.

Therefore, when designing the API, we need to reserve expandable interfaces to facilitate the subsequent support for more other inference frameworks.

# Design Details
## Overall Architecture

<div style="text-align: center;"> <img src="images/new-inference-mode.png" alt="new-inference-mode.jpg" style="zoom:33%;" /> </div>

**Cloud/Edge Inference:** Inference instances at the cloud side and the edge side, with the inference services directly provided by the Triton framework (meanwhile, expansion capabilities are reserved to facilitate the subsequent expansion of other inference frameworks, such as vLLM, SGLang, LMDeploy...). 

**Side Car:** It is responsible for providing an external access entrance, intercepting user requests, and the corresponding difficult case mining algorithm is also implemented here, so as to decide whether the cloud side needs to intervene in the inference request. (The reason for not making the Sidecar an independent Agent is that the difficult case mining algorithms corresponding to different inference instances may vary, and standardization is not possible.) 

- The difficult case mining algorithm needs to be designed to be user-customizable, such as through functional computing?

**Edge/Cloud Model:** The inference models at the cloud side and the edge side can be expanded on the existing Model API, and need to support large language models. This model will be mounted in the inference instance. 

**Local Controller:** It is responsible for collecting the status information of the inference instances at the cloud side and the edge side, and cooperates with the GM to complete the lifecycle management of the entire inference task. 

**Global Manager:** The control plane of Sedna.

## Two Mode
<div style="text-align: center;"> <img src="images/llminference-mode-select.png" alt="llminference-mode-select.png" style="zoom:33%;" /> </div>

**SDK mode:** This mode is compatible with Sedna's original SDK mode and is suitable for users who want to customize their business directly in the inference container. It offers higher flexibility in usage, but requires a certain level of development experience. To use this mode, users need to be familiar with the development documentation of Sedna's [SDK](https://sedna.readthedocs.io/en/latest/api/lib/index.html) and deploy the image compiled based on their self-developed algorithms.

**One-click mode:** This mode is suitable for users who are not familiar with the SDK and do not want to use the SDK. Users only need to configure the model and the inference instance, and then they can start the inference service. The system will automatically create the corresponding inference instance and deploy it to the cloud side or the edge side.


## bacth deployment

The current joint inference paradigm of Sedna forcibly sets the inference instances at the cloud side and the edge side to be one. This approach is not conducive to the batch deployment scenario. 

Therefore, in our design, we have added support for the batch deployment of the same task. For the same task type, we support users to deploy multiple instances to multiple nodes simultaneously, while the cloud service reuses the same service, as shown in the following figure.

<div style="text-align: center;"> <img src="images/batch-inference.png" alt="batch-inference.png" style="zoom:33%;" /> </div>

**Some assumptions:**

**Cloud:**

When deploying models on the cloud, there is no need to specify nodes. Considering that large language models may be too large to run on a single node, parallel strategies such as model Tensor Parallelism (TP) or Pipeline Parallelism (PP) need to be adopted. Therefore, on the cloud, it may also be necessary to leverage the capabilities of Persistent Volume Claim (PVC) to achieve model sharing among various nodes.

**Edge:**

Multiple nodes can be specified simultaneously on the edge to meet batch deployment requirements. However, due to the complexity of the edge environment (such as unstable network and limited node resources...), we stipulate that small models on the edge should meet the basic conditions for running on a single node.

# implementation
## API Design
```go
type InferenceEngine string

const (
  Triton InferenceEngine = "triton"
  VLLM   InferenceEngine = "vllm"
  SGLang InferenceEngine = "sglang"
  // Add other engines as needed
)

const (
  // deployment mode
  DeploymentModeSdk      string = "sdk"
  DeploymentModeOneClick string = "oneclick"
)

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +kubebuilder:resource:shortName=llmji
// +kubebuilder:subresource:status

// LLMJointInferenceService describes the data that a llmjointinferenceservice resource should have
type LLMJointInferenceService struct {
  metav1.TypeMeta `json:",inline"`

  metav1.ObjectMeta `json:"metadata"`

  Spec   LLMJointInferenceServiceSpec   `json:"spec"`
  Status LLMJointInferenceServiceStatus `json:"status,omitempty"`
}

// LLMJointInferenceServiceSpec is a description of a LLMjointinferenceservice
type LLMJointInferenceServiceSpec struct {
  // DeploymentMode is the deployment mode
  // +kubebuilder:validation:Enum=sdk;oneclick
  // +enum
  DeploymentMode string `json:"deploymentMode"`

  // InferenceConfig is the configuration for the inference engine
  // +kubebuilder:validation:Required
  InferenceConfig InferenceConfig `json:"inferenceConfig"`

  // EdgeWorker is the worker configuration for edge deployment
  // +kubebuilder:validation:Optional
  EdgeWorker LLMEdgeWorker `json:"edgeWorker"`

  // CloudWorker is the worker configuration for cloud deployment
  // +kubebuilder:validation:Optional
  CloudWorker LLMCloudWorker `json:"cloudWorker"`
}

// InferenceEngine describes the inference engine,
// it will be used to create a ConfigMap, which is then injected into the Pod instance through a volume.
type InferenceConfig struct {
  // InferenceEngine describes the inference engine
  // +kubebuilder:validation:Enum=triton;vllm;sglang
  // +enum
  Engine InferenceEngine `json:"engine"`

  // TritonConfig describes the triton config
  // +kubebuilder:validation:Optional
  TritonConfig *TritonConfig `json:"tritonConfig,omitempty"`
  // VLLMConfig   *VLLMConfig     `json:"vllmConfig,omitempty"`
  // SGLangConfig *SGLangConfig   `json:"sglangConfig,omitempty"`
  // Add other engines as needed
}

// TritonConfig describes the triton config
type TritonConfig struct {
  // todo
}

// EdgeWorker describes the data a edge worker should have
type LLMEdgeWorker struct {
  // mode is the model to be used
  // +kubebuilder:validation:Required
  // +kubebuilder:validation:MinItems=1
  Nodes []string `json:"nodes,omitempty"`

  // model is the model to be usedo
  // +kubebuilder:validation:Required
  Model LLMSmallModel `json:"model"`

  // HardExampleMining is the hard example mining algorithm to be used
  // +kubebuilder:validation:Optional
  HardExampleMining LLMHardExampleMining `json:"hardExampleMining"`

  // template is the pod template
  // +kubebuilder:validation:Required
  Template v1.PodTemplateSpec `json:"template"`
}

// CloudWorker describes the data a cloud worker should have
type LLMCloudWorker struct {
  // mode is the model to be used
  // +kubebuilder:validation:Required
  Model LLMBigModel `json:"model"`

  // template is the pod template
  // +kubebuilder:validation:Required
  Template v1.PodTemplateSpec `json:"template"`
}

// SmallModel describes the small model
type LLMSmallModel struct {
  // ModelName is the name of the model
  // +kubebuilder:validation:Required
  Name string `json:"name"`
}

// BigModel describes the big model
type LLMBigModel struct {
  // ModelName is the name of the model
  // +kubebuilder:validation:Required
  Name string `json:"name"`
}

// HardExampleMining describes the hard example algorithm to be used
type LLMHardExampleMining struct {
  // Algorithm is the name of the algorithm
  // +kubebuilder:validation:Required
  Name string `json:"name"`

  // Parameters is the parameters of the algorithm
  // +kubebuilder:validation:Optional
  Parameters []ParaSpec `json:"parameters,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// LLMJointInferenceServiceList is a list of LLMJointInferenceServices.
type LLMJointInferenceServiceList struct {
  metav1.TypeMeta `json:",inline"`
  metav1.ListMeta `json:"metadata"`
  Items           []LLMJointInferenceService `json:"items"`
}

// LLMJointInferenceServiceStatus represents the current state of a joint inference service.
type LLMJointInferenceServiceStatus struct {

  // The latest available observations of a joint inference service's current state.
  // +optional
  Conditions []LLMJointInferenceServiceCondition `json:"conditions,omitempty"`

  // Represents time when the service was acknowledged by the service controller.
  // It is not guaranteed to be set in happens-before order across separate operations.
  // It is represented in RFC3339 form and is in UTC.
  // +optional
  StartTime *metav1.Time `json:"startTime,omitempty"`

  // The number of actively running workers.
  // +optional
  Active int32 `json:"active"`

  // The number of workers which reached to Failed.
  // +optional
  Failed int32 `json:"failed"`

  // Metrics of the joint inference service.
  Metrics []Metric `json:"metrics,omitempty"`
}

// LLMJointInferenceServiceConditionType defines the condition type
type LLMJointInferenceServiceConditionType string

// These are valid conditions of a service.
const (
  // LLMJointInferenceServiceCondPending means the service has been accepted by the system,
  // but one or more of the workers has not been started.
  LLMJointInferenceServiceCondPending LLMJointInferenceServiceConditionType = "Pending"
  // LLMJointInferenceServiceCondFailed means the service has failed its execution.
  LLMJointInferenceServiceCondFailed LLMJointInferenceServiceConditionType = "Failed"
  // LLMJointInferenceServiceCondRunning means the service is running.
  LLMJointInferenceServiceCondRunning LLMJointInferenceServiceConditionType = "Running"
)

// LLMJointInferenceServiceCondition describes current state of a service.
// see https://github.com/kubernetes/community/blob/master/contributors/devel/sig-architecture/api-conventions.md#typical-status-properties for details.
type LLMJointInferenceServiceCondition struct {
  // Type of service condition, Complete or Failed.
  Type LLMJointInferenceServiceConditionType `json:"type"`
  // Status of the condition, one of True, False, Unknown.
  Status v1.ConditionStatus `json:"status"`
  // last time we got an update on a given condition
  // +optional
  LastHeartbeatTime metav1.Time `json:"lastHeartbeatTime,omitempty"`
  // Last time the condition transit from one status to another.
  // +optional
  LastTransitionTime metav1.Time `json:"lastTransitionTime,omitempty"`
  // (brief) reason for the condition's last transition,
  // one-word CamelCase reason for the condition's last transition.
  // +optional
  Reason string `json:"reason,omitempty"`
  // Human readable message indicating details about last transition.
  // +optional
  Message string `json:"message,omitempty"`
}
```
- Use ConfigMap to save the images of each inference engine.
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: inference-engine-config
data:
  triton: "tritonserver:21.06"    # Triton Inference engine image
  vllm: "vllm:latest"             # VLLM Inference engine image
  sglang: "sglang:latest"         # SGLang Inference engine image
```

### example
```yaml
apiVersion: example.com/v1
kind: LLMJointInferenceService
metadata:
  name: my-llm-joint-inference-service
spec:
  inferenceConfig:
    engine: triton
    tritonConfig:
      ...
  edgeWorker:
    nodes:
      - "edge-node-1"
      - "edge-node-2"
    model:
      name: "edge-small-model"
    hardExampleMining:
      name: "hard-example-mining-algorithm"
      parameters:
        - name: "min-magnitude"
          value: "0.5"
        - name: "batch-size"
          value: "32"
    resource:
      requests:
        cpu: "500m"
        memory: "1Gi"
      limits:
        cpu: "2"
        memory: "4Gi"
  cloudWorker:
    model:
      name: "cloud-large-model"
    resource:
      requests:
        cpu: "1"
        memory: "8Gi"
      limits:
        cpu: "4"
        memory: "16Gi"
```

### Development Progress Planning

#### Phase 1
- Triton API design
- Ability to deploy discriminative models using the new paradigm (Triton)
- Compatibility with the old SDK
- Development of a hard case mining sidecar
- Provision of examples

#### Phase 2
- Implementation of the vLLM API
- Support for deploying large language models using vLLM
- Support for the hard case mining algorithm for vLLM large language models
- Provision of examples

#### Phase 3
- Implementation of the Ollama API
- Support for deploying large language models using Ollama
- Support for the hard case mining algorithm for Ollama large language models
- Provision of examples

#### Phase 4
- Implementation of the LMDeploy API
- Support for deploying large language models using LMDeploy
- Support for the hard case mining algorithm for LMDeploy large language models
- Provision of examples
