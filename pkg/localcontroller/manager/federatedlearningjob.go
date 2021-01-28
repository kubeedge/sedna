package manager

import (
	"encoding/json"

	"k8s.io/klog/v2"

	sednav1 "github.com/kubeedge/sedna/pkg/apis/sedna/v1alpha1"
	"github.com/kubeedge/sedna/pkg/localcontroller/db"
	"github.com/kubeedge/sedna/pkg/localcontroller/gmclient"
	"github.com/kubeedge/sedna/pkg/localcontroller/util"
)

// FederatedLearningManager defines federated-learning-job manager
type FederatedLearningManager struct {
	Client               gmclient.ClientI
	WorkerMessageChannel chan WorkerMessage
}

// FederatedLearning defines config for federated-learning-job
type FederatedLearning struct {
	*sednav1.FederatedLearningJob
}

const (
	//FederatedLearningJobKind is kind of federated-learning-job resource
	FederatedLearningJobKind = "federatedlearningjob"
)

// NewFederatedLearningManager creates a federated-learning-job manager
func NewFederatedLearningManager(client gmclient.ClientI) FeatureManager {
	fm := &FederatedLearningManager{
		Client:               client,
		WorkerMessageChannel: make(chan WorkerMessage, WorkerMessageChannelCacheSize),
	}

	return fm
}

// Start starts federated-learning-job manager
func (fm *FederatedLearningManager) Start() error {
	go fm.monitorWorker()

	return nil
}

// monitorWorker monitors message from worker
func (fm *FederatedLearningManager) monitorWorker() {
	for {
		workerMessageChannel := fm.WorkerMessageChannel
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
func (fm *FederatedLearningManager) Insert(message *gmclient.Message) error {
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
func (fm *FederatedLearningManager) Delete(message *gmclient.Message) error {
	name := util.GetUniqueIdentifier(message.Header.Namespace, message.Header.ResourceName, message.Header.ResourceKind)
	if err := db.DeleteResource(name); err != nil {
		return err
	}

	return nil
}

// AddWorkerMessage adds worker messages to the channel
func (fm *FederatedLearningManager) AddWorkerMessage(message WorkerMessage) {
	fm.WorkerMessageChannel <- message
}

// GetName returns the name of the manager
func (fm *FederatedLearningManager) GetName() string {
	return FederatedLearningJobKind
}
