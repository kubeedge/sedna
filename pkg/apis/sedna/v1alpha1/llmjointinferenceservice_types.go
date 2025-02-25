/*
Copyright 2025 The KubeEdge Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package v1alpha1

import (
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

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
