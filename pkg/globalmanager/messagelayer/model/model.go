package model

type MessageHeader struct {
	Namespace string `json:"namespace"`

	ResourceKind string `json:"resourceKind"`

	ResourceName string `json:"resourceName"`

	Operation string `json:"operation"`
}

// Message struct
type Message struct {
	MessageHeader `json:"header"`
	Content       []byte `json:"content"`
}
