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
	batchv1 "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +kubebuilder:resource:shortName=reid
// +kubebuilder:subresource:status

type ReidJob struct {
	metav1.TypeMeta `json:",inline"`

	metav1.ObjectMeta `json:"metadata"`

	Spec   ReidJobSpec   `json:"spec"`
	Status ReidJobStatus `json:"status,omitempty"`
}

// ReidJobSpec is a description of a ReidJob
type ReidJobSpec struct {
	batchv1.JobSpec `json:",inline"`
	KafkaSupport    bool `json:"kafkaSupport,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ReidJobList is a list of ReidJob.
type ReidJobList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`
	Items           []ReidJob `json:"items"`
}

// ReidJobStatus represents the current state of a reid job
type ReidJobStatus struct {
	// The latest available observations of a reid job's current state.
	// +optional
	Conditions []ReidJobCondition `json:"conditions,omitempty"`

	// Represents time when the job was acknowledged by the job controller.
	// It is not guaranteed to be set in happens-before order across separate operations.
	// It is represented in RFC3339 form and is in UTC.
	// +optional
	StartTime *metav1.Time `json:"startTime,omitempty"`

	// Represents time when the job was completed. It is not guaranteed to
	// be set in happens-before order across separate operations.
	// It is represented in RFC3339 form and is in UTC.
	// +optional
	CompletionTime *metav1.Time `json:"completionTime,omitempty"`

	// The number of actively running pods.
	// +optional
	Active int32 `json:"active"`

	// The number of pods which reached phase Succeeded.
	// +optional
	Succeeded int32 `json:"succeeded"`

	// The number of pods which reached phase Failed.
	// +optional
	Failed int32 `json:"failed"`

	// The phase of the reid job.
	// +optional
	Phase ReidJobPhase `json:"phase,omitempty"`
}

type ReidJobConditionType string

// These are valid stage conditions of a job.
const (
	ReidJobCondWaiting   ReidJobConditionType = "Waiting"
	ReidJobCondReady     ReidJobConditionType = "Ready"
	ReidJobCondStarting  ReidJobConditionType = "Starting"
	ReidJobCondRunning   ReidJobConditionType = "Running"
	ReidJobCondCompleted ReidJobConditionType = "Completed"
	ReidJobCondFailed    ReidJobConditionType = "Failed"
)

// ReidJobCondition describes current state of a job.
// see https://github.com/kubernetes/community/blob/master/contributors/devel/sig-architecture/api-conventions.md#typical-status-properties for details.
type ReidJobCondition struct {
	// Type of job condition, Complete or Failed.
	Type ReidJobConditionType `json:"type"`
	// Status of the condition, one of True, False, Unknown.
	Status v1.ConditionStatus `json:"status"`
	// last time we got an update on a given condition
	// +optional
	LastHeartbeatTime metav1.Time `json:"lastHeartbeatTime,omitempty"`
	// Last time the condition transit from one status to another.
	// +optional
	LastTransitionTime metav1.Time `json:"lastTransitionTime,omitempty"`
	// (brief) reason for the condition's last transition.
	// +optional
	Reason string `json:"reason,omitempty"`
	// Human readable message indicating details about last transition.
	// +optional
	Message string `json:"message,omitempty"`
	// The json data related to this condition
	// +optional
	Data string `json:"data,omitempty"`
}

// ReidJobPhase is a label for the condition of a job at the current time.
type ReidJobPhase string

// These are the valid statuses of jobs.
const (
	// ReidJobPending means the job has been accepted by the system, but one or more of the pods
	// has not been started. This includes time before being bound to a node, as well as time spent
	// pulling images onto the host.
	ReidJobPending ReidJobPhase = "Pending"
	// ReidJobRunning means the job has been bound to a node and all of the pods have been started.
	// At least one container is still running or is in the process of being restarted.
	ReidJobRunning ReidJobPhase = "Running"
	// ReidJobSucceeded means that all pods in the job have voluntarily terminated
	// with a container exit code of 0, and the system is not going to restart any of these pods.
	ReidJobSucceeded ReidJobPhase = "Succeeded"
	// ReidJobFailed means that all pods in the job have terminated, and at least one container has
	// terminated in a failure (exited with a non-zero exit code or was stopped by the system).
	ReidJobFailed ReidJobPhase = "Failed"
)
