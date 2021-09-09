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

// DNNPartitioningManager defines dnn-partitioning-service manager
type DNNPartitioningManager struct {
	Client               gmclient.ClientI
	WorkerMessageChannel chan WorkerMessage
}

const (
	// DNNPartitioningServiceKind is kind of dnn-partitioning-service resource
	DNNPartitioningServiceKind = "dnnpartitioningservice"
)

// NewDNNPartitioningManager creates a dnn partitioning manager
func NewDNNPartitioningManager(client gmclient.ClientI) FeatureManager {
	jm := &DNNPartitioningManager{
		Client:               client,
		WorkerMessageChannel: make(chan WorkerMessage, WorkerMessageChannelCacheSize),
	}

	return jm
}

// Start starts dnn-partitioning-service manager
func (jm *DNNPartitioningManager) Start() error {
	go jm.monitorWorker()

	return nil
}

// monitorWorker monitors message from worker
func (jm *DNNPartitioningManager) monitorWorker() {
	for {
		workerMessageChannel := jm.WorkerMessageChannel
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

		if err := jm.Client.WriteMessage(um, header); err != nil {
			klog.Errorf("dnn-partitioning-service(name=%s) uploads worker(name=%s) message failed, error: %v",
				name, workerMessage.Name, err)
		}
	}
}

// Insert inserts jdnn-partitioning-service config in db
func (jm *DNNPartitioningManager) Insert(message *gmclient.Message) error {
	name := util.GetUniqueIdentifier(message.Header.Namespace, message.Header.ResourceName, message.Header.ResourceKind)

	ji := sednav1.DNNPartitioningService{}

	if err := json.Unmarshal(message.Content, &ji); err != nil {
		return err
	}

	if err := db.SaveResource(name, ji.TypeMeta, ji.ObjectMeta, ji.Spec); err != nil {
		return err
	}

	return nil
}

// Delete deletes dnn-partitioning-service config in db
func (jm *DNNPartitioningManager) Delete(message *gmclient.Message) error {
	name := util.GetUniqueIdentifier(message.Header.Namespace, message.Header.ResourceName, message.Header.ResourceKind)
	if err := db.DeleteResource(name); err != nil {
		return err
	}

	return nil
}

// AddWorkerMessage adds worker messages
func (jm *DNNPartitioningManager) AddWorkerMessage(message WorkerMessage) {
	jm.WorkerMessageChannel <- message
}

// GetName gets kind of the manager
func (jm *DNNPartitioningManager) GetName() string {
	return DNNPartitioningServiceKind
}
