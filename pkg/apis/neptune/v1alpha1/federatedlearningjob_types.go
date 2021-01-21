package v1alpha1

import (
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
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
}

// AggregationWorker describes the data an aggregation worker should have
type AggregationWorker struct {
	Model      modelRefer            `json:"model"`
	NodeName   string                `json:"nodeName"`
	WorkerSpec AggregationWorkerSpec `json:"workerSpec"`
}

// TrrainingWorker describes the data a training worker should have
type TrainingWorker struct {
	NodeName   string             `json:"nodeName"`
	WorkerSpec TrainingWorkerSpec `json:"workerSpec"`
	Dataset    datasetRefer       `json:"dataset"`
}

// AggregationWorkerSpec is a description of a aggregationworker
type AggregationWorkerSpec struct {
	CommonWorkerSpec
}

// TrainingWorkerSpec is a description of a trainingworker
type TrainingWorkerSpec struct {
	CommonWorkerSpec
}

type datasetRefer struct {
	Name string `json:"name"`
}

type modelRefer struct {
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
	// FLJobComplete means the job has completed its execution.
	FLJobCondComplete FLJobConditionType = "Complete"
	// FLJobFailed means the job has failed its execution.
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
