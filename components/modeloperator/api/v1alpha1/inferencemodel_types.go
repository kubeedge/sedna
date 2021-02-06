/*


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

// InferenceModelSpec defines the desired state of an InferenceModel.
// Two ways of deployment are provided. The first is through a docker image.
// The second is through a data store, with a manifest of model files.
// If image is provided, manifest and targetVersion will be ignored.
type InferenceModelSpec struct {
	ModelName     string `json:"modelName"`
	DeployToLayer string `json:"deployToLayer"`
	FrameworkType string `json:"frameworkType"`

	// +optional
	NodeSelector map[string]string `json:"nodeSelector,omitempty"`

	// +optional
	NodeName string `json:"nodeName,omitempty"`

	// +optional
	Image string `json:"image,omitempty"`

	// +optional
	Manifest []InferenceModelFile `json:"manifest,omitempty"`

	// +optional
	TargetVersion string `json:"targetVersion,omitempty"`

	// +optional
	// +kubebuilder:validation:Minimum=0
	ServingPort int32 `json:"servingPort"`

	// +optional
	// +kubebuilder:validation:Minimum=0
	Replicas *int32 `json:"replicas,omitempty"`
}

// InferenceModelFile defines an archive file for a single version of the model
type InferenceModelFile struct {
	Version     string `json:"version,omitempty"`
	DownloadURL string `json:"downloadURL,omitempty"`
	Sha256sum   string `json:"sha256sum,omitempty"`
}

// InferenceModelStatus defines the observed state of InferenceModel
type InferenceModelStatus struct {
	URL            string `json:"url,omitempty"`
	ServingVersion string `json:"servingVersion,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status

// InferenceModel is the Schema for the inferencemodels API.
// For cloud, it deploys a single deployment and a service.
// For edge, it deploys a single pod.
type InferenceModel struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   InferenceModelSpec   `json:"spec,omitempty"`
	Status InferenceModelStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// InferenceModelList contains a list of InferenceModel
type InferenceModelList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []InferenceModel `json:"items"`
}

func init() {
	SchemeBuilder.Register(&InferenceModel{}, &InferenceModelList{})
}
