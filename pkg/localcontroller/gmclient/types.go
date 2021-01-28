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
