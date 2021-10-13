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

package manager

import (
	"encoding/json"

	"k8s.io/klog/v2"

	sednav1 "github.com/kubeedge/sedna/pkg/apis/sedna/v1alpha1"
	"github.com/kubeedge/sedna/pkg/localcontroller/db"
	"github.com/kubeedge/sedna/pkg/localcontroller/gmclient"
	"github.com/kubeedge/sedna/pkg/localcontroller/util"
)

// JointInferenceManager defines joint-inference-service manager
type MultiEdgeTrackingManager struct {
	Client               gmclient.ClientI
	WorkerMessageChannel chan WorkerMessage
}

const (
	// MultiEdgeTrackingService is kind of joint-inference-service resource
	MultiEdgeTrackingServiceKind = "multiedgetrackingservice"
)

// NewJointInferenceManager creates a joint inference manager
func NewMultiEdgeTrackingManager(client gmclient.ClientI) FeatureManager {
	mm := &MultiEdgeTrackingManager{
		Client:               client,
		WorkerMessageChannel: make(chan WorkerMessage, WorkerMessageChannelCacheSize),
	}

	return mm
}

// Start starts joint-inference-service manager
func (mm *MultiEdgeTrackingManager) Start() error {
	go mm.monitorWorker()

	return nil
}

// monitorWorker monitors message from worker
func (mm *MultiEdgeTrackingManager) monitorWorker() {
	for {
		workerMessageChannel := mm.WorkerMessageChannel
		workerMessage, ok := <-workerMessageChannel
		if !ok {
			break
		}

		name := util.GetUniqueIdentifier(workerMessage.Namespace, workerMessage.OwnerName, workerMessage.OwnerKind)
		header := gmclient.MessageHeader{
			Namespace:    workerMessage.Namespace,
			ResourceKind: workerMessage.OwnerKind,
			ResourceName: workerMessage.OwnerName,
			Operation:    gmclient.StatusOperation,
		}

		um := UpstreamMessage{
			Phase:  workerMessage.Kind,
			Status: workerMessage.Status,
			Output: &WorkerOutput{
				OwnerInfo: workerMessage.OwnerInfo,
			},
		}

		if err := mm.Client.WriteMessage(um, header); err != nil {
			klog.Errorf("MOT service(name=%s) uploads worker(name=%s) message failed, error: %v",
				name, workerMessage.Name, err)
		}
	}
}

// Insert inserts joint-inference-service config in db
func (mm *MultiEdgeTrackingManager) Insert(message *gmclient.Message) error {
	name := util.GetUniqueIdentifier(message.Header.Namespace, message.Header.ResourceName, message.Header.ResourceKind)

	ms := sednav1.MultiEdgeTrackingService{}

	if err := json.Unmarshal(message.Content, &ms); err != nil {
		return err
	}

	if err := db.SaveResource(name, ms.TypeMeta, ms.ObjectMeta, ms.Spec); err != nil {
		return err
	}

	return nil
}

// Delete deletes joint-inference-service config in db
func (mm *MultiEdgeTrackingManager) Delete(message *gmclient.Message) error {
	name := util.GetUniqueIdentifier(message.Header.Namespace, message.Header.ResourceName, message.Header.ResourceKind)
	if err := db.DeleteResource(name); err != nil {
		return err
	}

	return nil
}

// AddWorkerMessage adds worker messages
func (mm *MultiEdgeTrackingManager) AddWorkerMessage(message WorkerMessage) {
	mm.WorkerMessageChannel <- message
}

// GetName gets kind of the manager
func (mm *MultiEdgeTrackingManager) GetName() string {
	return MultiEdgeTrackingServiceKind
}
