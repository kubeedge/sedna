package manager

import (
	"encoding/json"

	"k8s.io/klog/v2"

	"github.com/edgeai-neptune/neptune/pkg/localcontroller/db"
	"github.com/edgeai-neptune/neptune/pkg/localcontroller/util"
	"github.com/edgeai-neptune/neptune/pkg/localcontroller/wsclient"
)

// FederatedLearningManager defines federated-learning-job manager
type FederatedLearningManager struct {
	Client               *wsclient.Client
	WorkerMessageChannel chan WorkerMessage
}

// FederatedLearning defines config for federated-learning-job
type FederatedLearning struct {
	APIVersion string                 `json:"apiVersion"`
	Kind       string                 `json:"kind"`
	MetaData   map[string]interface{} `json:"metadata"`
	Spec       map[string]interface{} `json:"spec"`
}

const (
	//FederatedLearningJobKind is kind of federated-learning-job resource
	FederatedLearningJobKind = "federatedlearningjob"
)

// NewFederatedLearningManager creates a federated-learning-job manager
func NewFederatedLearningManager(client *wsclient.Client) FeatureManager {
	fm := &FederatedLearningManager{
		Client:               client,
		WorkerMessageChannel: make(chan WorkerMessage, WorkerMessageChannelCacheSize),
	}

	return fm
}

// Start starts federated-learning-job manager
func (fm *FederatedLearningManager) Start() error {
	if err := fm.Client.Subscribe(FederatedLearningJobKind, fm.handleMessage); err != nil {
		klog.Errorf("register federated-learning-job manager to the client failed, error: %v", err)
		return err
	}

	go fm.monitorWorker()

	klog.Infof("start federated-learning-job manager successfully")

	return nil
}

// handleMessage handles the message from GlobalManager
func (fm *FederatedLearningManager) handleMessage(message *wsclient.Message) {
	uniqueIdentifier := util.GetUniqueIdentifier(message.Header.Namespace, message.Header.ResourceName, message.Header.ResourceKind)
	switch message.Header.Operation {
	case InsertOperation:
		if err := fm.insertJob(uniqueIdentifier, message.Content); err != nil {
			klog.Errorf("insert %s(name=%s) to db failed, error: %v", message.Header.ResourceKind, uniqueIdentifier, err)
		}

	case DeleteOperation:
		if err := fm.deleteJob(uniqueIdentifier); err != nil {
			klog.Errorf("delete %s(name=%s) to db failed, error: %v", message.Header.ResourceKind, uniqueIdentifier, err)
		}
	}
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
		header := wsclient.MessageHeader{
			Namespace:    workerMessage.Namespace,
			ResourceKind: workerMessage.OwnerKind,
			ResourceName: workerMessage.OwnerName,
			Operation:    StatusOperation,
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

// insertJob inserts federated-learning-job config in db
func (fm *FederatedLearningManager) insertJob(name string, payload []byte) error {
	federatedLearning := FederatedLearning{}

	if err := json.Unmarshal(payload, &federatedLearning); err != nil {
		return err
	}

	metaData, err := json.Marshal(federatedLearning.MetaData)
	if err != nil {
		return err
	}

	spec, err := json.Marshal(federatedLearning.Spec)
	if err != nil {
		return err
	}

	r := db.Resource{
		Name:       name,
		APIVersion: federatedLearning.APIVersion,
		Kind:       federatedLearning.Kind,
		MetaData:   string(metaData),
		Spec:       string(spec),
	}

	if err = db.SaveResource(&r); err != nil {
		return err
	}

	return nil
}

// deleteJob deletes federated-learning-job config in db
func (fm *FederatedLearningManager) deleteJob(name string) error {
	if err := db.DeleteResource(name); err != nil {
		return err
	}

	return nil
}

// AddWorkerMessageToChannel adds worker messages to the channel
func (fm *FederatedLearningManager) AddWorkerMessageToChannel(message WorkerMessage) {
	fm.WorkerMessageChannel <- message
}

// GetKind gets kind of the manager
func (fm *FederatedLearningManager) GetKind() string {
	return FederatedLearningJobKind
}
