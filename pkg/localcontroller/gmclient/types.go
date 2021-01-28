package gmclient

const (
	// InsertOperation is the insert value
	InsertOperation = "insert"
	// DeleteOperation is the delete value
	DeleteOperation = "delete"
	// StatusOperation is the status value
	StatusOperation = "status"
)

// Message defines message
type Message struct {
	Header  MessageHeader `json:"header"`
	Content []byte        `json:"content"`
}

// MessageHeader define header of message
type MessageHeader struct {
	Namespace    string `json:"namespace"`
	ResourceKind string `json:"resourceKind"`
	ResourceName string `json:"resourceName"`
	Operation    string `json:"operation"`
}

type MessageResourceHandler interface {
	GetName() string
	Insert(*Message) error
	Delete(*Message) error
}

type ClientI interface {
	Start() error
	WriteMessage(messageBody interface{}, messageHeader MessageHeader) error
	Subscribe(m MessageResourceHandler) error
}
