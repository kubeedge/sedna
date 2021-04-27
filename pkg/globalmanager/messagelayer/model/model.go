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

package model

import "fmt"

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

func (m *Message) String() string {
	return fmt.Sprintf("MessageHeader: %+v, Content: +%s", m.MessageHeader, string(m.Content))
}
