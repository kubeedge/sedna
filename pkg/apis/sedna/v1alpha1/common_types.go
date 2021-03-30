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

// Metric describes the data that a resource model metric should have
type Metric struct {
	Key   string `json:"key"`
	Value string `json:"value"`
}

// ParaSpec is a description of a parameter
type ParaSpec struct {
	Key   string `json:"key"`
	Value string `json:"value"`
}
