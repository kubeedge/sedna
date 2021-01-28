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
type JointInferenceManager struct {
	Client               gmclient.ClientI
	WorkerMessageChannel chan WorkerMessage
}

const (
	// JointInferenceServiceKind is kind of joint-inference-service resource
	JointInferenceServiceKind = "jointinferenceservice"
)

// NewJointInferenceManager creates a joint inference manager
func NewJointInferenceManager(client gmclient.ClientI) FeatureManager {
	jm := &JointInferenceManager{
		Client:               client,
		WorkerMessageChannel: make(chan WorkerMessage, WorkerMessageChannelCacheSize),
	}

	return jm
}

// Start starts joint-inference-service manager
func (jm *JointInferenceManager) Start() error {
	go jm.monitorWorker()

	return nil
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
			klog.Errorf("joint-inference-service(name=%s) uploads worker(name=%s) message failed, error: %v",
				name, workerMessage.Name, err)
		}
	}
}

// Insert inserts joint-inference-service config in db
func (jm *JointInferenceManager) Insert(message *gmclient.Message) error {
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

// Delete deletes joint-inference-service config in db
func (jm *JointInferenceManager) Delete(message *gmclient.Message) error {
	name := util.GetUniqueIdentifier(message.Header.Namespace, message.Header.ResourceName, message.Header.ResourceKind)
	if err := db.DeleteResource(name); err != nil {
		return err
	}

	return nil
}

// AddWorkerMessage adds worker messages
func (jm *JointInferenceManager) AddWorkerMessage(message WorkerMessage) {
	jm.WorkerMessageChannel <- message
}

// GetName gets kind of the manager
func (jm *JointInferenceManager) GetName() string {
	return JointInferenceServiceKind
}
