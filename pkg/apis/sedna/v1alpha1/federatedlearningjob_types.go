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
// +kubebuilder:resource:shortName=fl
// +kubebuilder:subresource:status

// FederatedLearningJob describes the data that a FederatedLearningJob resource should have
type FederatedLearningJob struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   FLJobSpec   `json:"spec"`
	Status FLJobStatus `json:"status,omitempty"`
}

// FLJobSpec is a description of a federatedlearning job
type FLJobSpec struct {
	AggregationWorker AggregationWorker `json:"aggregationWorker"`
	TrainingWorkers   []TrainingWorker  `json:"trainingWorkers"`
	PretrainedModel   PretrainedModel   `json:"pretrainedModel,omitempty"`
	Transmitter       Transmitter       `json:"transmitter,omitempty"`
}

// Transmitter describes the transmitter of data plane between training workers and aggregation worker
type Transmitter struct {
	S3 *S3Transmitter `json:"s3,omitempty"`
	WS *WSTransmitter `json:"ws,omitempty"`
}

// S3Transmitter describes the s3 transmitter
type S3Transmitter struct {
	AggregationDataPath string `json:"aggDataPath"`
	CredentialName      string `json:"credentialName,omitempty"`
}

// WSTransmitter describes the websocket transmitter
type WSTransmitter struct{}

// AggregationWorker describes the data an aggregation worker should have
type AggregationWorker struct {
	// Model defines train model of federated learning job
	Model    TrainModel         `json:"model"`
	Template v1.PodTemplateSpec `json:"template"`
}

// TrainingWorker describes the data a training worker should have
type TrainingWorker struct {
	Dataset  TrainDataset       `json:"dataset"`
	Template v1.PodTemplateSpec `json:"template"`
}

// TrainDataset defines dataset of federated learning job
type TrainDataset struct {
	Name string `json:"name"`
}

type TrainModel struct {
	Name string `json:"name"`
}

// PretrainedModel defines pretrained model of federated learning job
type PretrainedModel struct {
	Name string `json:"name"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// FederatedLearningJobList is a list of FederatedLearningJobs.
type FederatedLearningJobList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`
	Items           []FederatedLearningJob `json:"items"`
}

// FLJobStatus represents the current state of a federatedlearning job.
type FLJobStatus struct {

	// The latest available observations of a federated job's current state.
	// +optional
	Conditions []FLJobCondition `json:"conditions,omitempty"`

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

	// The phase of the federatedlearning job.
	// +optional
	Phase FLJobPhase `json:"phase,omitempty"`
}

type FLJobConditionType string

// These are valid conditions of a job.
const (
	// FLJobCondComplete means the job has completed its execution.
	FLJobCondComplete FLJobConditionType = "Complete"
	// FLJobCondFailed means the job has failed its execution.
	FLJobCondFailed FLJobConditionType = "Failed"
	// FLJobCondTraining means the job has been training.
	FLJobCondTraining FLJobConditionType = "Training"
)

// FLJobCondition describes current state of a job.
type FLJobCondition struct {
	// Type of job condition, Complete or Failed.
	Type FLJobConditionType `json:"type"`
	// Status of the condition, one of True, False, Unknown.
	Status v1.ConditionStatus `json:"status"`
	// Last time the condition was checked.
	// +optional
	LastProbeTime metav1.Time `json:"lastProbeTime,omitempty"`
	// Last time the condition transit from one status to another.
	// +optional
	LastHeartbeatTime metav1.Time `json:"lastHeartbeatTime,omitempty"`
	// (brief) reason for the condition's last transition.
	// +optional
	Reason string `json:"reason,omitempty"`
	// Human readable message indicating details about last transition.
	// +optional
	Message string `json:"message,omitempty"`
}

// FLJobPhase is a label for the condition of a job at the current time.
type FLJobPhase string

// These are the valid statuses of jobs.
const (
	// FLJobPending means the job has been accepted by the system, but one or more of the pods
	// has not been started. This includes time before being bound to a node, as well as time spent
	// pulling images onto the host.
	FLJobPending FLJobPhase = "Pending"
	// FLJobRunning means the job has been bound to a node and all of the pods have been started.
	// At least one container is still running or is in the process of being restarted.
	FLJobRunning FLJobPhase = "Running"
	// FLJobSucceeded means that all pods in the job have voluntarily terminated
	// with a container exit code of 0, and the system is not going to restart any of these pods.
	FLJobSucceeded FLJobPhase = "Succeeded"
	// FLJobFailed means that all pods in the job have terminated, and at least one container has
	// terminated in a failure (exited with a non-zero exit code or was stopped by the system).
	FLJobFailed FLJobPhase = "Failed"
)
