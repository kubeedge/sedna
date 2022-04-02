package v1alpha1

//go:generate controller-gen object paths=$GOFILE

import (
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type IncrementalLearningJob struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec ILJobSpec `json:"spec"`
	Status ILJobStatus `json:"status,omitempty"`
}
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type IncrementalLearningJobList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`

	Items []IncrementalLearningJob `json:"items"`
}

// ILJobSpec is a description of a incrementallearningjob
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type ILJobSpec struct {
	Dataset      ILDataset    `json:"dataset"`
	InitialModel InitialModel `json:"initialModel"`
	TrainSpec    TrainSpec    `json:"trainSpec"`
	EvalSpec     EvalSpec     `json:"evalSpec"`
	DeploySpec   DeploySpec   `json:"deploySpec"`

	// the credential referer for OutputDir
	CredentialName string `json:"credentialName,omitempty"`
	OutputDir      string `json:"outputDir"`
}

// TrainSpec describes the data an train worker should have
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type TrainSpec struct {
	Template v1.PodTemplateSpec `json:"template"`
	Trigger  Trigger            `json:"trigger"`
}

// EvalSpec describes the data an eval worker should have
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type EvalSpec struct {
	Template v1.PodTemplateSpec `json:"template"`
}

// DeploySpec describes the deploy model to be updated
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type DeploySpec struct {
	Model             DeployModel        `json:"model"`
	Trigger           Trigger            `json:"trigger"`
	HardExampleMining HardExampleMining  `json:"hardExampleMining"`
	Template          v1.PodTemplateSpec `json:"template"`
}
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type Trigger struct {
	CheckPeriodSeconds int       `json:"checkPeriodSeconds,omitempty"`
	Timer              *Timer    `json:"timer,omitempty"`
	Condition          Condition `json:"condition"`
}
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type Timer struct {
	Start string `json:"start"`
	End   string `json:"end"`
}
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type Condition struct {
	Operator  string  `json:"operator"`
	Threshold float64 `json:"threshold"`
	Metric    string  `json:"metric"`
}
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type ILDataset struct {
	Name      string  `json:"name"`
	TrainProb float64 `json:"trainProb"`
}
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type InitialModel struct {
	Name string `json:"name"`
}
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type DeployModel struct {
	Name string `json:"name"`
	// HotUpdateEnabled will enable the model hot update feature if its value is true.
	// Default value is false.
	HotUpdateEnabled bool `json:"hotUpdateEnabled,omitempty"`
	// PollPeriodSeconds is interval in seconds between echo poll of the deploy model config file.
	// PollPeriodSeconds must be greater than zero and the default value is 60.
	// It will be used only when HotUpdateEnabled is true.
	// +kubebuilder:validation:Minimum:=1
	// +kubebuilder:default:=60
	PollPeriodSeconds int64 `json:"pollPeriodSeconds,omitempty"`
}

// HardExampleMining describes the hard example algorithm to be used
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type HardExampleMining struct {
	Name       string     `json:"name"`
	Parameters []ParaSpec `json:"parameters,omitempty"`
}

// ParaSpec is a description of a parameter
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type ParaSpec struct {
	Key   string `json:"key"`
	Value string `json:"value"`
}
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
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
// ILJobCondition describes current state of a job.
// see https://github.com/kubernetes/community/blob/master/contributors/devel/sig-architecture/api-conventions.md#typical-status-properties for details.
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
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
