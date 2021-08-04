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

package federatedlearning

import (
	"encoding/json"

	"k8s.io/klog/v2"

	sednav1 "github.com/kubeedge/sedna/pkg/apis/sedna/v1alpha1"
	"github.com/kubeedge/sedna/pkg/localcontroller/db"
	clienttypes "github.com/kubeedge/sedna/pkg/localcontroller/gmclient"
	types "github.com/kubeedge/sedna/pkg/localcontroller/managers"
	"github.com/kubeedge/sedna/pkg/localcontroller/util"
	workertypes "github.com/kubeedge/sedna/pkg/localcontroller/worker"
)

// FederatedLearningManager defines federated-learning-job manager
type Manager struct {
	Client               clienttypes.ClientI
	WorkerMessageChannel chan workertypes.MessageContent
}

// FederatedLearning defines config for federated-learning-job
type FederatedLearning struct {
	*sednav1.FederatedLearningJob
}

const (
	//KindName is kind of federated-learning-job resource
	KindName = "federatedlearningjob"
)

// New creates a federated-learning-job types
func New(client clienttypes.ClientI) types.FeatureManager {
	fm := &Manager{
		Client:               client,
		WorkerMessageChannel: make(chan workertypes.MessageContent, workertypes.MessageChannelCacheSize),
	}

	return fm
}

// Start starts federated-learning-job manager
func (fm *Manager) Start() error {
	go fm.monitorWorker()

	return nil
}

// monitorWorker monitors message from worker
func (fm *Manager) monitorWorker() {
	for {
		workerMessageChannel := fm.WorkerMessageChannel
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
				Models:    workerMessage.Results,
				OwnerInfo: workerMessage.OwnerInfo,
			},
		}

		if err := fm.Client.WriteMessage(um, header); err != nil {
			klog.Errorf("federated-learning-job(name=%s) uploads worker(name=%s) message failed, error: %v",
				name, workerMessage.Name, err)
		}
	}
}

// Insert inserts federated-learning-job config in db
func (fm *Manager) Insert(message *clienttypes.Message) error {
	name := util.GetUniqueIdentifier(message.Header.Namespace, message.Header.ResourceName, message.Header.ResourceKind)

	fl := FederatedLearning{}

	if err := json.Unmarshal(message.Content, &fl); err != nil {
		return err
	}

	if err := db.SaveResource(name, fl.TypeMeta, fl.ObjectMeta, fl.Spec); err != nil {
		return err
	}

	return nil
}

// Delete deletes federated-learning-job config in db
func (fm *Manager) Delete(message *clienttypes.Message) error {
	name := util.GetUniqueIdentifier(message.Header.Namespace, message.Header.ResourceName, message.Header.ResourceKind)
	if err := db.DeleteResource(name); err != nil {
		return err
	}

	return nil
}

// AddWorkerMessage adds worker messages to the channel
func (fm *Manager) AddWorkerMessage(message workertypes.MessageContent) {
	fm.WorkerMessageChannel <- message
}

// GetName returns the name of the manager
func (fm *Manager) GetName() string {
	return KindName
}
