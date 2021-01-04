package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Dataset describes the data that a dataset resource should have
type Dataset struct {
	metav1.TypeMeta `json:",inline"`

	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   DatasetSpec   `json:"spec"`
	Status DatasetStatus `json:"status"`
}

// DatasetSpec is a description of a dataset
type DatasetSpec struct {
	URL      string `json:"url"`
	Format   string `json:"format"`
	NodeName string `json:"nodeName"`
}

// DatasetStatus represents information about the status of a dataset
// including the time a dataset updated, and number of samples in a dataset
type DatasetStatus struct {
	UpdateTime      *metav1.Time `json:"updateTime,omitempty" protobuf:"bytes,1,opt,name=updateTime"`
	NumberOfSamples int          `json:"numberOfSamples"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// DatasetList is a list of Datasets
type DatasetList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`

	Items []Dataset `json:"items"`
}
