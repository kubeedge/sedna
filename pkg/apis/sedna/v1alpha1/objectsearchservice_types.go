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
// +kubebuilder:resource:shortName=objs
// +kubebuilder:subresource:status

// ObjectSearchService describes the data that a objectsearchservice resource should have
type ObjectSearchService struct {
	metav1.TypeMeta `json:",inline"`

	metav1.ObjectMeta `json:"metadata"`

	Spec   ObjectSearchServiceSpec   `json:"spec"`
	Status ObjectSearchServiceStatus `json:"status,omitempty"`
}

// ObjectSearchServiceSpec is a description of a objectsearchservice
type ObjectSearchServiceSpec struct {
	UserWorker      UserWorker       `json:"userWorker"`
	TrackingWorkers []TrackingWorker `json:"trackingWorkers"`
	ReidWorkers     ReidWorkers      `json:"reidWorkers"`
}

// UserWorker describes the data a user worker should have
type UserWorker struct {
	Template v1.PodTemplateSpec `json:"template"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ObjectSearchServiceList is a list of ObjectSearchServices.
type ObjectSearchServiceList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`
	Items           []ObjectSearchService `json:"items"`
}

// ObjectSearchServiceStatus represents the current state of a object search service.
type ObjectSearchServiceStatus struct {

	// The latest available observations of a object search service's current state.
	// +optional
	Conditions []ObjectSearchServiceCondition `json:"conditions,omitempty"`

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
}

// ObjectSearchServiceConditionType defines the condition type
type ObjectSearchServiceConditionType string

// These are valid conditions of a service.
const (
	// ObjectSearchServiceCondPending means the service has been accepted by the system,
	// but one or more of the workers has not been started.
	ObjectSearchServiceCondPending ObjectSearchServiceConditionType = "Pending"
	// ObjectSearchServiceCondFailed means the service has failed its execution.
	ObjectSearchServiceCondFailed ObjectSearchServiceConditionType = "Failed"
	// ObjectSearchServiceCondRunning means the service is running.
	ObjectSearchServiceCondRunning ObjectSearchServiceConditionType = "Running"
)

// ObjectSearchServiceCondition describes current state of a service.
// see https://github.com/kubernetes/community/blob/master/contributors/devel/sig-architecture/api-conventions.md#typical-status-properties for details.
type ObjectSearchServiceCondition struct {
	// Type of service condition, Complete or Failed.
	Type ObjectSearchServiceConditionType `json:"type"`
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
