/*
Copyright 2021 The KubeEdge Authors.

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

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +kubebuilder:resource:shortName=dp
// +kubebuilder:subresource:status

// DNNPartitioningService describes the data that a dnnpartitioningservice resource should have
type DNNPartitioningService struct {
	metav1.TypeMeta `json:",inline"`

	metav1.ObjectMeta `json:"metadata"`

	Spec   DNNPartitioningServiceSpec   `json:"spec"`
	Status DNNPartitioningServiceStatus `json:"status,omitempty"`
}

// DNNPartitioningServiceSpec is a description of a dnnpartitioningservice
type DNNPartitioningServiceSpec struct {
	DNNPartitioningEdgeWorker  []DNNPartitioningEdgeWorker `json:"dnnpartitioningedgeWorker"`
	DNNPartitioningCloudWorker DNNPartitioningCloudWorker  `json:"dnnpartitioningcloudWorker"`
}

// DNNPartitioningEdgeWorker describes the data a edge worker should have
type DNNPartitioningEdgeWorker struct {
	Model             EdgeModel         `json:"model"`
	Template          v1.PodTemplateSpec `json:"template"`
}

// DNNPartitioningCloudWorker describes the data a cloud worker should have
type DNNPartitioningCloudWorker struct {
	Model    CloudModel           `json:"model"`
	Template v1.PodTemplateSpec `json:"template"`
}

// EdgeModel describes the edge model
type EdgeModel struct {
	Name string `json:"name"`
}

// CloudModel describes the cloud model
type CloudModel struct {
	Name string `json:"name"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// DNNPartitioningServiceList is a list of DNNPartitioningServices.
type DNNPartitioningServiceList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`
	Items           []DNNPartitioningService `json:"items"`
}

// DNNPartitioningServiceStatus represents the current state of a DNN Partitioning service.
type DNNPartitioningServiceStatus struct {

	// The latest available observations of a DNN Partitioning service's current state.
	// +optional
	Conditions []DNNPartitioningServiceCondition `json:"conditions,omitempty"`

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

	// Metrics of the DNN Partitioning service.
	Metrics []Metric `json:"metrics,omitempty"`
}

// DNNPartitioningServiceConditionType defines the condition type
type DnnPartitioningServiceConditionType string

// These are valid conditions of a service.
const (
	// DNNPartitioningServiceCondPending means the service has been accepted by the system,
	// but one or more of the workers has not been started.
	DNNPartitioningServiceCondPending DnnPartitioningServiceConditionType = "Pending"
	// DNNPartitioningServiceCondFailed means the service has failed its execution.
	DNNPartitioningServiceCondFailed DnnPartitioningServiceConditionType = "Failed"
	// DNNPartitioningServiceReady means the service has been ready.
	DNNPartitioningServiceCondRunning DnnPartitioningServiceConditionType = "Running"
)

// DNNPartitioningServiceCondition describes current state of a service.
// see https://github.com/kubernetes/community/blob/master/contributors/devel/sig-architecture/api-conventions.md#typical-status-properties for details.
type DNNPartitioningServiceCondition struct {
	// Type of service condition, Complete or Failed.
	Type DnnPartitioningServiceConditionType `json:"type"`
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
