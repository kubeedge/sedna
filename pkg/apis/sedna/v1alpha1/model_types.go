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

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +kubebuilder:subresource:status

// Model describes the data that a model resource should have
type Model struct {
	metav1.TypeMeta `json:",inline"`

	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   ModelSpec   `json:"spec"`
	Status ModelStatus `json:"status,omitempty"`
}

// ModelSpec is a description of a model
type ModelSpec struct {
	URL    string `json:"url"`
	Format string `json:"format"`

	// The following are optional fields for the model
	CredentialName string `json:"credentialName,omitempty"`
	// +optional
	Description string `json:"description,omitmepty"`
	// +optional
	Purpose string `json:"purpose,omitmepty"`
	// +optional
	Classes []string `json:"classes,omitmepty"`
	// +optional
	Extra []ExtraVar `json:"extra,omitempty"`
}

// ModelStatus represents information about the status of a model
// including the time a model updated, and metrics in a model
type ModelStatus struct {
	UpdateTime *metav1.Time `json:"updateTime,omitempty" protobuf:"bytes,1,opt,name=updateTime"`
	Metrics    []Metric     `json:"metrics,omitempty" protobuf:"bytes,2,rep,name=metrics"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ModelList is a list of Models
type ModelList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`

	Items []Model `json:"items"`
}

// ExtraVar represents an extra variable associated with the model.
type ExtraVar struct {
	// Name of the environment variable. Must be a C_IDENTIFIER.
	Name  string `json:"name" protobuf:"bytes,1,opt,name=name"`
	Value string `json:"value,omitempty" protobuf:"bytes,2,opt,name=value"`
}
