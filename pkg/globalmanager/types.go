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

package globalmanager

import (
	"encoding/json"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// WorkerPara describes the system-defined parameters of worker
type WorkerPara struct {
	volumeMountList []string
	volumeList      []string
	volumeMapName   []string
	env             map[string]string
	workerType      string
	// if true, force to use hostNetwork
	hostNetwork bool

	restartPolicy v1.RestartPolicy
}

// CommonInterface describes the commom interface of CRs
type CommonInterface interface {
	metav1.Object
	schema.ObjectKind
	runtime.Object
}

// FeatureControllerI defines the interface of an AI Feature controller
type FeatureControllerI interface {
	Start() error
	GetName() string
}

type Model struct {
	Format  string                 `json:"format,omitempty"`
	URL     string                 `json:"url,omitempty"`
	Metrics map[string]interface{} `json:"metrics,omitempty"`
}

// the data of this condition including the input/output to do the next step
type IncrementalCondData struct {
	Input *struct {
		// Only one model cases
		Model  *Model  `json:"model,omitempty"`
		Models []Model `json:"models,omitempty"`

		DataURL   string `json:"dataURL,omitempty"`
		OutputDir string `json:"outputDir,omitempty"`
	} `json:"input,omitempty"`

	Output *struct {
		Model  *Model  `json:"model,omitempty"`
		Models []Model `json:"models,omitempty"`
	} `json:"output,omitempty"`
}

func (m *Model) GetURL() string {
	return m.URL
}

func (cd *IncrementalCondData) joinModelURLs(model *Model, models []Model) []string {
	var modelURLs []string
	if model != nil {
		modelURLs = append(modelURLs, model.GetURL())
	} else {
		for _, m := range models {
			modelURLs = append(modelURLs, m.GetURL())
		}
	}
	return modelURLs
}

func (cd *IncrementalCondData) GetInputModelURLs() []string {
	return cd.joinModelURLs(cd.Input.Model, cd.Input.Models)
}

func (cd *IncrementalCondData) GetOutputModelURLs() []string {
	return cd.joinModelURLs(cd.Output.Model, cd.Output.Models)
}

func (cd *IncrementalCondData) Unmarshal(data []byte) error {
	return json.Unmarshal(data, cd)
}

func (cd IncrementalCondData) Marshal() ([]byte, error) {
	return json.Marshal(cd)
}
