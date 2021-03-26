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
// +kubebuilder:resource:shortName=il
// +kubebuilder:subresource:status

// IncrementalLearningJob describes the data that a incrementallearningjob resource should have
type IncrementalLearningJob struct {
	metav1.TypeMeta `json:",inline"`

	metav1.ObjectMeta `json:"metadata"`

	Spec   ILJobSpec   `json:"spec"`
	Status ILJobStatus `json:"status,omitempty"`
}

// ILJobSpec is a description of a incrementallearningjob
type ILJobSpec struct {
	Dataset      ILDataset    `json:"dataset"`
	OutputDir    string       `json:"outputDir"`
	InitialModel InitialModel `json:"initialModel"`
	TrainSpec    TrainSpec    `json:"trainSpec"`
	EvalSpec     EvalSpec     `json:"evalSpec"`
	DeploySpec   DeploySpec   `json:"deploySpec"`
}

// TrainSpec describes the data an train worker should have
type TrainSpec struct {
	Template v1.PodTemplateSpec `json:"template"`
	Trigger  Trigger            `json:"trigger"`
}

// EvalSpec describes the data an eval worker should have
type EvalSpec struct {
	Template v1.PodTemplateSpec `json:"template"`
}

// DeploySpec describes the deploy model to be updated
type DeploySpec struct {
	Model             DeployModel        `json:"model"`
	Trigger           Trigger            `json:"trigger"`
	HardExampleMining HardExampleMining  `json:"hardExampleMining"`
	Template          v1.PodTemplateSpec `json:"template"`
}

type Trigger struct {
	CheckPeriodSeconds int       `json:"checkPeriodSeconds,omitempty"`
	Timer              *Timer    `json:"timer,omitempty"`
	Condition          Condition `json:"condition"`
}

type Timer struct {
	Start string `json:"start"`
	End   string `json:"end"`
}

type Condition struct {
	Operator  string  `json:"operator"`
	Threshold float64 `json:"threshold"`
	Metric    string  `json:"metric"`
}

type ILDataset struct {
	Name      string  `json:"name"`
	TrainProb float64 `json:"trainProb"`
}

type InitialModel struct {
	Name string `json:"name"`
}

type DeployModel struct {
	Name string `json:"name"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// IncrementalLearningJobList is a list of IncrementalLearningJobs.
type IncrementalLearningJobList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`
	Items           []IncrementalLearningJob `json:"items"`
}

// ILJobStatus represents the current state of a incrementallearning job
type ILJobStatus struct {
	// The latest available observations of a incrementllearning job's current state.
	// +optional
	Conditions []ILJobCondition `json:"conditions,omitempty"`

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
}

type ILJobStageConditionType string

// These are valid stage conditions of a job.
const (
	ILJobStageCondWaiting   ILJobStageConditionType = "Waiting"
	ILJobStageCondReady     ILJobStageConditionType = "Ready"
	ILJobStageCondStarting  ILJobStageConditionType = "Starting"
	ILJobStageCondRunning   ILJobStageConditionType = "Running"
	ILJobStageCondCompleted ILJobStageConditionType = "Completed"
	ILJobStageCondFailed    ILJobStageConditionType = "Failed"
)

// ILJobCondition describes current state of a job.
// see https://github.com/kubernetes/community/blob/master/contributors/devel/sig-architecture/api-conventions.md#typical-status-properties for details.
type ILJobCondition struct {
	// Type of job condition, Complete or Failed.
	Type ILJobStageConditionType `json:"type"`
	// Status of the condition, one of True, False, Unknown.
	Status v1.ConditionStatus `json:"status"`
	// Stage of the condition
	Stage ILJobStage `json:"stage"`
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

// ILJobStage is a label for the stage of a job at the current time.
type ILJobStage string

const (
	ILJobTrain  ILJobStage = "Train"
	ILJobEval   ILJobStage = "Eval"
	ILJobDeploy ILJobStage = "Deploy"
)
