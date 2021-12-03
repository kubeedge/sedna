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

package multiedgetracking

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

const (
	KindName = "multiedgetrackingservice"
)

// New creates a joint inference manager
func New(client clienttypes.ClientI) types.FeatureManager {
	jm := &Manager{
		Client:               client,
		WorkerMessageChannel: make(chan workertypes.MessageContent, workertypes.MessageChannelCacheSize),
	}

	return jm
}

// JointInferenceManager defines multi-edge-tracking-service manager
type Manager struct {
	Client               clienttypes.ClientI
	WorkerMessageChannel chan workertypes.MessageContent
}

// Start starts multi-edge-tracking-service manager
func (mm *Manager) Start() error {
	go mm.monitorWorker()

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
			klog.Errorf("multi-edge-tracking-service(name=%s) uploads worker(name=%s) message failed, error: %v",
				name, workerMessage.Name, err)
		}
	}
}

// Insert inserts multi-edge-tracking-service config in db
func (jm *Manager) Insert(message *clienttypes.Message) error {
	name := util.GetUniqueIdentifier(message.Header.Namespace, message.Header.ResourceName, message.Header.ResourceKind)

	ji := sednav1.JointInferenceService{}

	if err := json.Unmarshal(message.Content, &ji); err != nil {
		return err
	}

	if err := db.SaveResource(name, ji.TypeMeta, ji.ObjectMeta, ji.Spec); err != nil {
		return err
	}

	return nil
}

// Delete deletes multi-edge-tracking-service config in db
func (jm *Manager) Delete(message *clienttypes.Message) error {
	name := util.GetUniqueIdentifier(message.Header.Namespace, message.Header.ResourceName, message.Header.ResourceKind)
	if err := db.DeleteResource(name); err != nil {
		return err
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
