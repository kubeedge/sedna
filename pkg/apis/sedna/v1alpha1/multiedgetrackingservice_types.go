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
	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +kubebuilder:resource:shortName=mi
// +kubebuilder:subresource:status

type MultiEdgeTrackingService struct {
	metav1.TypeMeta `json:",inline"`

	metav1.ObjectMeta `json:"metadata"`

	Spec   MultiEdgeTrackingServiceSpec   `json:"spec"`
	Status MultiEdgeTrackingServiceStatus `json:"status,omitempty"`
}

// MultiEdgeTrackingSpec is a description of a MultiEdgeTracking
type MultiEdgeTrackingServiceSpec struct {
	MultiObjectTrackingDeploy MultiObjectTrackingDeploy `json:"motDeploy"`
	FEDeploy                  FEDeploy                  `json:"feDeploy"`
	ReIDDeploy                ReIDDeploy                `json:"reIDDeploy"`
}

// EdgeWorker describes the data a edge worker should have
type MultiObjectTrackingDeploy struct {
	Spec appsv1.DeploymentSpec `json:"spec"`
}

type FEDeploy struct {
	Spec appsv1.DeploymentSpec `json:"spec"`
}

type ReIDDeploy struct {
	Spec appsv1.DeploymentSpec `json:"spec"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// MultiEdgeTrackingList is a list of MultiEdgeTrackings.
type MultiEdgeTrackingServiceList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`
	Items           []MultiEdgeTrackingService `json:"items"`
}

// MultiEdgeTrackingStatus represents the current state of a joint inference service.
type MultiEdgeTrackingServiceStatus struct {

	// The latest available observations of a joint inference service's current state.
	// +optional
	Conditions []MultiEdgeTrackingServiceCondition `json:"conditions,omitempty"`

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

// MultiEdgeTrackingConditionType defines the condition type
type MultiEdgeTrackingServiceConditionType string

// These are valid conditions of a service.
const (
	// MultiEdgeTrackingCondPending means the service has been accepted by the system,
	// but one or more of the workers has not been started.
	MultiEdgeTrackingServiceCondPending MultiEdgeTrackingServiceConditionType = "Pending"
	// MultiEdgeTrackingCondFailed means the service has failed its execution.
	MultiEdgeTrackingServiceCondFailed MultiEdgeTrackingServiceConditionType = "Failed"
	// MultiEdgeTrackingReady means the service has been ready.
	MultiEdgeTrackingServiceCondRunning MultiEdgeTrackingServiceConditionType = "Running"
)

// MultiEdgeTrackingCondition describes current state of a service.
// see https://github.com/kubernetes/community/blob/master/contributors/devel/sig-architecture/api-conventions.md#typical-status-properties for details.
type MultiEdgeTrackingServiceCondition struct {
	// Type of service condition, Complete or Failed.
	Type MultiEdgeTrackingServiceConditionType `json:"type"`
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
