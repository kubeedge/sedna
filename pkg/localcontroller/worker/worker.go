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

package worker

// MessageContent defines the body content of message that comes from workers.
type MessageContent struct {
	// Name is worker name
	Name      string `json:"name"`
	Namespace string `json:"namespace"`
	// OwnerName is job name
	OwnerName string `json:"ownerName"`
	// OwnerKind is kind of job
	OwnerKind string `json:"ownerKind"`
	// OwnerInfo is info about job
	OwnerInfo map[string]interface{} `json:"ownerInfo"`
	// Kind is worker phase, include train/eval/deploy
	Kind string `json:"kind"`
	// Status is worker status, include running/completed/failed
	Status string `json:"status"`
	// Results is the output of worker when it was completed
	Results []map[string]interface{} `json:"results"`
}

const (
	// MessageChannelCacheSize is size of worker message channel cache
	MessageChannelCacheSize = 100

	// ReadyStatus is the ready status about worker
	ReadyStatus = "ready"
	// CompletedStatus is the completed status about worker
	CompletedStatus = "completed"
	// FailedStatus is the failed status about worker
	FailedStatus = "failed"
)
