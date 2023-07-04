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

import (
	messagetypes "github.com/kubeedge/sedna/pkg/globalmanager/messagelayer/model"
	"github.com/kubeedge/sedna/pkg/globalmanager/runtime"
)

const (
	// InsertOperation is the insert value
	InsertOperation = "insert"
	// DeleteOperation is the delete value
	DeleteOperation = "delete"
	// StatusOperation is the status value
	StatusOperation = "status"
)

type Model = runtime.Model

// Message defines message between LC and GM
type Message struct {
	Header  MessageHeader `json:"header"`
	Content []byte        `json:"content"`
}

// MessageHeader defines the header between LC and GM
type MessageHeader = messagetypes.MessageHeader

// UpstreamMessage defines send message content to GM
type UpstreamMessage struct {
	Phase  string  `json:"phase"`
	Status string  `json:"status"`
	Input  *Input  `json:"input,omitempty"`
	Output *Output `json:"output"`
}

type Input struct {
	Models       []Model `json:"models,omitempty"`
	DataURL      string  `json:"dataURL,omitempty"`
	DataIndexURL string  `json:"dataIndexURL,omitempty"`
	OutputDir    string  `json:"outputDir,omitempty"`
}

type Output struct {
	// All the TaskInfo has been written to "Model"
	Models    []map[string]interface{} `json:"models"`
	OwnerInfo map[string]interface{}   `json:"ownerInfo"`
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
