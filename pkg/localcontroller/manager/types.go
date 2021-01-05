package manager

const (
	// WorkerMessageChannelCacheSize is size of channel cache
	WorkerMessageChannelCacheSize = 100

	InsertOperation = "insert"
	DeleteOperation = "delete"
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

// UpstreamMessage defines send message to GlobalManager
type UpstreamMessage struct {
	Phase  string        `json:"phase"`
	Status string        `json:"status"`
	Output *WorkerOutput `json:"output"`
}

// WorkerOutput defines output information of worker
type WorkerOutput struct {
	Models    []map[string]interface{} `json:"models"`
	OwnerInfo map[string]interface{}   `json:"ownerInfo"`
}

// FeatureManager defines feature manager
type FeatureManager interface {
	Start() error
	GetKind() string
	AddWorkerMessageToChannel(message WorkerMessage)
}
