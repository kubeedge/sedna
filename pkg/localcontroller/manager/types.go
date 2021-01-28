package manager

import (
	"github.com/edgeai-neptune/neptune/pkg/localcontroller/gmclient"
)

const (
	// WorkerMessageChannelCacheSize is size of channel cache
	WorkerMessageChannelCacheSize = 100

	// TrainPhase is the train phase in incremental-learning-job
	TrainPhase = "train"
	// EvalPhase is the eval phase in incremental-learning-job
	EvalPhase = "eval"
	// DeployPhase is the deploy phase in incremental-learning-job
	DeployPhase = "deploy"

	// WorkerWaitingStatus is the waiting status about worker
	WorkerWaitingStatus = "waiting"

	// WorkerReadyStatus is the ready status about worker
	WorkerReadyStatus = "ready"
	// WorkerCompletedStatus is the completed status about worker
	WorkerCompletedStatus = "completed"
	// WorkerFailedStatus is the failed status about worker
	WorkerFailedStatus = "failed"

	// TriggerReadyStatus is the ready status about trigger in incremental-learning-job
	TriggerReadyStatus = "ready"
	// TriggerCompletedStatus is the completed status about trigger in incremental-learning-job
	TriggerCompletedStatus = "completed"
)

// WorkerMessage defines message struct from worker
type WorkerMessage struct {
	Name      string                   `json:"name"`
	Namespace string                   `json:"namespace"`
	OwnerName string                   `json:"ownerName"`
	OwnerKind string                   `json:"ownerKind"`
	Kind      string                   `json:"kind"`
	Status    string                   `json:"status"`
	OwnerInfo map[string]interface{}   `json:"ownerInfo"`
	Results   []map[string]interface{} `json:"results"`
}

// MetaData defines metadata
type MetaData struct {
	Name      string `json:"name"`
	Namespace string `json:"namespace"`
}

// ModelInfo defines model
type ModelInfo struct {
	Format  string               `json:"format"`
	URL     string               `json:"url"`
	Metrics map[string][]float64 `json:"metrics,omitempty"`
}

// UpstreamMessage defines send message to GlobalManager
type UpstreamMessage struct {
	Phase  string        `json:"phase"`
	Status string        `json:"status"`
	Input  *WorkerInput  `json:"input,omitempty"`
	Output *WorkerOutput `json:"output"`
}

type WorkerInput struct {
	// Only one model cases
	Models []ModelInfo `json:"models,omitempty"`

	DataURL   string `json:"dataURL,omitempty"`
	OutputDir string `json:"outputDir,omitempty"`
}

// WorkerOutput defines output information of worker
type WorkerOutput struct {
	Models    []map[string]interface{} `json:"models"`
	OwnerInfo map[string]interface{}   `json:"ownerInfo"`
}

// FeatureManager defines feature manager
type FeatureManager interface {
	// Start starts the manager
	Start() error

	// GetName returns name of the manager
	GetName() string

	// AddWorkerMessage dispatch the worker message to manager
	AddWorkerMessage(message WorkerMessage)

	// Insert includes gm message creation/updation
	Insert(*gmclient.Message) error

	Delete(*gmclient.Message) error
}
