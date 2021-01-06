package manager

import (
	"encoding/json"

	"k8s.io/klog/v2"

	"github.com/edgeai-neptune/neptune/pkg/localcontroller/db"
	"github.com/edgeai-neptune/neptune/pkg/localcontroller/util"
	"github.com/edgeai-neptune/neptune/pkg/localcontroller/wsclient"
)

// JointInferenceManager defines joint-inference-service manager
type JointInferenceManager struct {
	Client               *wsclient.Client
	WorkerMessageChannel chan WorkerMessage
}

// JointInference defines config for joint-inference-service
type JointInference struct {
	APIVersion string                 `json:"apiVersion"`
	Kind       string                 `json:"kind"`
	MetaData   map[string]interface{} `json:"metadata"`
	Spec       map[string]interface{} `json:"spec"`
}

const (
	// JointInferenceServiceKind is kind of joint-inference-service resource
	JointInferenceServiceKind = "jointinferenceservice"
)

// NewJointInferenceManager creates a joint inference manager
func NewJointInferenceManager(client *wsclient.Client) FeatureManager {
	jm := &JointInferenceManager{
		Client:               client,
		WorkerMessageChannel: make(chan WorkerMessage, WorkerMessageChannelCacheSize),
	}

	return jm
}

// Start starts joint-inference-service manager
func (jm *JointInferenceManager) Start() error {
	if err := jm.Client.Subscribe(JointInferenceServiceKind, jm.handleMessage); err != nil {
		klog.Errorf("register joint-inference-service manager to the client failed, error: %v", err)
		return err
	}

	go jm.monitorWorker()

	klog.Infof("start joint-inference-service manager successfully")

	return nil
}

// handleMessage handles the message from GlobalManager
func (jm *JointInferenceManager) handleMessage(message *wsclient.Message) {
	uniqueIdentifier := util.GetUniqueIdentifier(message.Header.Namespace, message.Header.ResourceName, message.Header.ResourceKind)

	switch message.Header.Operation {
	case InsertOperation:
		if err := jm.insertService(uniqueIdentifier, message.Content); err != nil {
			klog.Errorf("insert %s(name=%s) to db failed, error: %v", message.Header.ResourceKind, uniqueIdentifier, err)
		}

	case DeleteOperation:
		if err := jm.deleteService(uniqueIdentifier); err != nil {
			klog.Errorf("delete %s(name=%s) to db failed, error: %v", message.Header.ResourceKind, uniqueIdentifier, err)
		}
	}
}

// monitorWorker monitors message from worker
func (jm *JointInferenceManager) monitorWorker() {
	for {
		workerMessageChannel := jm.WorkerMessageChannel
		workerMessage, ok := <-workerMessageChannel
		if !ok {
			break
		}

		name := util.GetUniqueIdentifier(workerMessage.Namespace, workerMessage.OwnerName, workerMessage.OwnerKind)
		header := wsclient.MessageHeader{
			Namespace:    workerMessage.Namespace,
			ResourceKind: workerMessage.OwnerKind,
			ResourceName: workerMessage.OwnerName,
			Operation:    "status",
		}

		um := UpstreamMessage{
			Phase:  workerMessage.Kind,
			Status: workerMessage.Status,
			Output: &WorkerOutput{
				OwnerInfo: workerMessage.OwnerInfo,
			},
		}

		if err := jm.Client.WriteMessage(um, header); err != nil {
			klog.Errorf("joint-inference-service(name=%s) uploads worker(name=%s) message failed, error: %v",
				name, workerMessage.Name, err)
		}
	}
}

// insertService inserts joint-inference-service config in db
func (jm *JointInferenceManager) insertService(name string, payload []byte) error {
	jointInference := JointInference{}

	if err := json.Unmarshal(payload, &jointInference); err != nil {
		return err
	}

	metaData, err := json.Marshal(jointInference.MetaData)
	if err != nil {
		return err
	}

	spec, err := json.Marshal(jointInference.Spec)
	if err != nil {
		return err
	}

	r := db.Resource{
		Name:       name,
		APIVersion: jointInference.APIVersion,
		Kind:       jointInference.Kind,
		MetaData:   string(metaData),
		Spec:       string(spec),
	}

	if err = db.SaveResource(&r); err != nil {
		return err
	}

	return nil
}

// deleteService deletes joint-inference-service config in db
func (jm *JointInferenceManager) deleteService(name string) error {
	if err := db.DeleteResource(name); err != nil {
		return err
	}

	return nil
}

// AddWorkerMessageToChannel adds worker messages to the channel
func (jm *JointInferenceManager) AddWorkerMessageToChannel(message WorkerMessage) {
	jm.WorkerMessageChannel <- message
}

// GetKind gets kind of the manager
func (jm *JointInferenceManager) GetKind() string {
	return JointInferenceServiceKind
}
