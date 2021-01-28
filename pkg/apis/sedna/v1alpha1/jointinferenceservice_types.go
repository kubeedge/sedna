package v1alpha1

import (
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// JointInferenceService describes the data that a jointinferenceservice resource should have
type JointInferenceService struct {
	metav1.TypeMeta `json:",inline"`

	metav1.ObjectMeta `json:"metadata"`

	Spec   JointInferenceServiceSpec   `json:"spec"`
	Status JointInferenceServiceStatus `json:"status,omitempty"`
}

// JointInferenceServiceSpec is a description of a jointinferenceservice
type JointInferenceServiceSpec struct {
	EdgeWorker  EdgeWorker  `json:"edgeWorker"`
	CloudWorker CloudWorker `json:"cloudWorker"`
}

// EdgeWorker describes the data a edge worker should have
type EdgeWorker struct {
	Model             SmallModel        `json:"model"`
	NodeName          string            `json:"nodeName"`
	HardExampleMining HardExampleMining `json:"hardExampleMining"`
	WorkerSpec        CommonWorkerSpec  `json:"workerSpec"`
}

// CloudWorker describes the data a cloud worker should have
type CloudWorker struct {
	Model      BigModel         `json:"model"`
	NodeName   string           `json:"nodeName"`
	WorkerSpec CommonWorkerSpec `json:"workerSpec"`
}

// SmallModel describes the small model
type SmallModel struct {
	Name string `json:"name"`
}

// BigModel describes the big model
type BigModel struct {
	Name string `json:"name"`
}

// HardExampleMining describes the hard example algorithm to be used
type HardExampleMining struct {
	Name       string     `json:"name"`
	Parameters []ParaSpec `json:"parameters"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// JointInferenceServiceList is a list of JointInferenceServices.
type JointInferenceServiceList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`
	Items           []JointInferenceService `json:"items"`
}

// JointInferenceServiceStatus represents the current state of a joint inference service.
type JointInferenceServiceStatus struct {

	// The latest available observations of a joint inference service's current state.
	// +optional
	Conditions []JointInferenceServiceCondition `json:"conditions,omitempty"`

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

// JointInferenceServiceConditionType defines the condition type
type JointInferenceServiceConditionType string

// These are valid conditions of a service.
const (
	// JointInferenceServiceCondPending means the service has been accepted by the system,
	// but one or more of the workers has not been started.
	JointInferenceServiceCondPending JointInferenceServiceConditionType = "Pending"
	// JointInferenceServiceCondFailed means the service has failed its execution.
	JointInferenceServiceCondFailed JointInferenceServiceConditionType = "Failed"
	// JointInferenceServiceReady means the service has been ready.
	JointInferenceServiceCondRunning JointInferenceServiceConditionType = "Running"
)

// JointInferenceServiceCondition describes current state of a service.
// see https://github.com/kubernetes/community/blob/master/contributors/devel/sig-architecture/api-conventions.md#typical-status-properties for details.
type JointInferenceServiceCondition struct {
	// Type of service condition, Complete or Failed.
	Type JointInferenceServiceConditionType `json:"type"`
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
