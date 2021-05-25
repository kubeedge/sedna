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
// +kubebuilder:resource:shortName=ll
// +kubebuilder:subresource:status

type LifelongLearningJob struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata"`
	Spec              LLJobSpec   `json:"spec"`
	Status            LLJobStatus `json:"status,omitempty"`
}

type LLJobSpec struct {
	Dataset    LLDataset    `json:"dataset"`
	TrainSpec  LLTrainSpec  `json:"trainSpec"`
	EvalSpec   LLEvalSpec   `json:"evalSpec"`
	DeploySpec LLDeploySpec `json:"deploySpec"`

	// the credential referer for OutputDir
	CredentialName string `json:"credentialName,omitempty"`
	OutputDir      string `json:"outputDir"`
}

type LLDataset struct {
	Name      string  `json:"name"`
	TrainProb float64 `json:"trainProb"`
}

// LLTrainSpec describes the data an train worker should have
type LLTrainSpec struct {
	Template v1.PodTemplateSpec `json:"template"`
	Trigger  LLTrigger          `json:"trigger"`
}

type LLTrigger struct {
	CheckPeriodSeconds int         `json:"checkPeriodSeconds,omitempty"`
	Timer              *LLTimer    `json:"timer,omitempty"`
	Condition          LLCondition `json:"condition"`
}

type LLTimer struct {
	Start string `json:"start"`
	End   string `json:"end"`
}

type LLCondition struct {
	Operator  string  `json:"operator"`
	Threshold float64 `json:"threshold"`
	Metric    string  `json:"metric"`
}

// LLEvalSpec describes the data an eval worker should have
type LLEvalSpec struct {
	Template v1.PodTemplateSpec `json:"template"`
}

// LLDeploySpec describes the deploy model to be updated
type LLDeploySpec struct {
	Template v1.PodTemplateSpec `json:"template"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// LifelongLearningJobList is a list of LifelongLearningJobs.
type LifelongLearningJobList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`
	Items           []LifelongLearningJob `json:"items"`
}

// LLJobStatus represents the current state of a lifelonglearning job
type LLJobStatus struct {
	// The latest available observations of a lifelonglearning job's current state.
	// +optional
	Conditions []LLJobCondition `json:"conditions,omitempty"`

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

type LLJobStageConditionType string

// These are valid stage conditions of a job.
const (
	LLJobStageCondWaiting   LLJobStageConditionType = "Waiting"
	LLJobStageCondReady     LLJobStageConditionType = "Ready"
	LLJobStageCondStarting  LLJobStageConditionType = "Starting"
	LLJobStageCondRunning   LLJobStageConditionType = "Running"
	LLJobStageCondCompleted LLJobStageConditionType = "Completed"
	LLJobStageCondFailed    LLJobStageConditionType = "Failed"
)

// LLJobCondition describes current state of a job.
// see https://github.com/kubernetes/community/blob/master/contributors/devel/sig-architecture/api-conventions.md#typical-status-properties for details.
type LLJobCondition struct {
	// Type of job condition, Complete or Failed.
	Type LLJobStageConditionType `json:"type"`
	// Status of the condition, one of True, False, Unknown.
	Status v1.ConditionStatus `json:"status"`
	// Stage of the condition
	Stage LLJobStage `json:"stage"`
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

// LLJobStage is a label for the stage of a job at the current time.
type LLJobStage string

const (
	LLJobTrain  LLJobStage = "Train"
	LLJobEval   LLJobStage = "Eval"
	LLJobDeploy LLJobStage = "Deploy"
)
