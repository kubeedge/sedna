/*
Copyright 2025 The KubeEdge Authors.

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

package llmjointinference

import (
	"encoding/json"
	"fmt"

	"k8s.io/klog/v2"

	sednav1 "github.com/kubeedge/sedna/pkg/apis/sedna/v1alpha1"
	"github.com/kubeedge/sedna/pkg/localcontroller/db"
	clienttypes "github.com/kubeedge/sedna/pkg/localcontroller/gmclient"
	types "github.com/kubeedge/sedna/pkg/localcontroller/managers"
	"github.com/kubeedge/sedna/pkg/localcontroller/util"
	workertypes "github.com/kubeedge/sedna/pkg/localcontroller/worker"
)

// Manager defines llm-joint-inference-service manager
type Manager struct {
	Client               clienttypes.ClientI
	WorkerMessageChannel chan workertypes.MessageContent
}

const (
	// KindName is kind of llm-joint-inference-service resource
	KindName = "llmjointinferenceservice"
)

// New creates a llm joint inference manager
func New(client clienttypes.ClientI) types.FeatureManager {
	jm := &Manager{
		Client:               client,
		WorkerMessageChannel: make(chan workertypes.MessageContent, workertypes.MessageChannelCacheSize),
	}

	return jm
}

// Start starts llm-joint-inference-service manager
func (jm *Manager) Start() error {
	go jm.monitorWorker()
	return nil
}

// monitorWorker monitors message from worker
func (jm *Manager) monitorWorker() {
	for {
		workerMessageChannel := jm.WorkerMessageChannel
		workerMessage, ok := <-workerMessageChannel
		if !ok {
			break
		}

		name := util.GetUniqueIdentifier(workerMessage.Namespace, workerMessage.OwnerName, workerMessage.OwnerKind)
		header := clienttypes.MessageHeader{
			Namespace:    workerMessage.Namespace,
			ResourceKind: workerMessage.OwnerKind,
			ResourceName: workerMessage.OwnerName,
			Operation:    clienttypes.StatusOperation,
		}

		um := clienttypes.UpstreamMessage{
			Phase:  workerMessage.Kind,
			Status: workerMessage.Status,
			Output: &clienttypes.Output{
				OwnerInfo: workerMessage.OwnerInfo,
			},
		}

		if err := jm.Client.WriteMessage(um, header); err != nil {
			klog.Errorf("llm-joint-inference-service(name=%s) uploads worker(name=%s) message failed, error: %v",
				name, workerMessage.Name, err)
		}
	}
}

// Insert inserts llm-joint-inference-service config in db
func (jm *Manager) Insert(message *clienttypes.Message) error {
	name := util.GetUniqueIdentifier(message.Header.Namespace, message.Header.ResourceName, message.Header.ResourceKind)

	ji := sednav1.LLMJointInferenceService{}
	if err := json.Unmarshal(message.Content, &ji); err != nil {
		return fmt.Errorf("failed to unmarshal LLMJointInferenceService: %v", err)
	}

	// Validate the inference engine configuration
	if err := validateInferenceConfig(&ji); err != nil {
		return fmt.Errorf("invalid inference configuration: %v", err)
	}

	if err := db.SaveResource(name, ji.TypeMeta, ji.ObjectMeta, ji.Spec); err != nil {
		return fmt.Errorf("failed to save LLMJointInferenceService to db: %v", err)
	}

	return nil
}

// Delete deletes llm-joint-inference-service config in db
func (jm *Manager) Delete(message *clienttypes.Message) error {
	name := util.GetUniqueIdentifier(message.Header.Namespace, message.Header.ResourceName, message.Header.ResourceKind)
	if err := db.DeleteResource(name); err != nil {
		return fmt.Errorf("failed to delete LLMJointInferenceService from db: %v", err)
	}

	return nil
}

// AddWorkerMessage adds worker messages
func (jm *Manager) AddWorkerMessage(message workertypes.MessageContent) {
	jm.WorkerMessageChannel <- message
}

// GetName gets kind of the manager
func (jm *Manager) GetName() string {
	return KindName
}

// validateInferenceConfig validates the inference engine configuration
func validateInferenceConfig(service *sednav1.LLMJointInferenceService) error {
	switch service.Spec.InferenceConfig.Engine {
	case sednav1.Triton:
		if service.Spec.InferenceConfig.TritonConfig == nil {
			return fmt.Errorf("triton config is required when using Triton engine")
		}
	case sednav1.VLLM:
		// Add VLLM config validation when implemented
		return fmt.Errorf("VLLM engine is not yet supported")
	case sednav1.SGLang:
		// Add SGLang config validation when implemented
		return fmt.Errorf("SGLang engine is not yet supported")
	default:
		return fmt.Errorf("unsupported inference engine: %s", service.Spec.InferenceConfig.Engine)
	}

	return nil
}
