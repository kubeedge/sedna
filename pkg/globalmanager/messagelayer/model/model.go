package model

// MessageHeader defines the header between LC and GM
type MessageHeader struct {
	Namespace string `json:"namespace"`

	ResourceKind string `json:"resourceKind"`

	ResourceName string `json:"resourceName"`

	Operation string `json:"operation"`
}

// Message defines the message between LC and GM
type Message struct {
	MessageHeader `json:"header"`
	Content       []byte `json:"content"`
}
