package manager

import (
	"encoding/json"

	"k8s.io/klog/v2"

	"github.com/edgeai-neptune/neptune/pkg/localcontroller/db"
	"github.com/edgeai-neptune/neptune/pkg/localcontroller/util"
	"github.com/edgeai-neptune/neptune/pkg/localcontroller/wsclient"
)

// ModelManager defines model manager
type ModelManager struct {
	Client          *wsclient.Client
	ModelChannelMap map[string]chan Model
}

// Model defines config for model
type Model struct {
	APIVersion string     `json:"apiVersion"`
	Kind       string     `json:"kind"`
	MetaData   *MetaData  `json:"metadata"`
	Spec       *ModelSpec `json:"spec"`
}

// ModelSpec defines model spec
type ModelSpec struct {
	Format string `json:"format"`
	URL    string `json:"url"`
}

const (
	//ModelChannelCacheSize is size of channel cache
	ModelChannelCacheSize = 100
	// DatasetResourceKind is kind of dataset resource
	ModelResourceKind = "model"
)

// NewModelManager creates a model manager
func NewModelManager(client *wsclient.Client) (*ModelManager, error) {
	mm := ModelManager{
		ModelChannelMap: make(map[string]chan Model),
		Client:          client,
	}

	if err := mm.initModelManager(); err != nil {
		klog.Errorf("init model manager failed, error: %v", err)
		return nil, err
	}

	return &mm, nil
}

// initModelManager inits model manager
func (mm *ModelManager) initModelManager() error {
	if err := mm.Client.Subscribe(ModelResourceKind, mm.handleMessage); err != nil {
		klog.Errorf("register model manager to the client failed, error: %v", err)
		return err
	}
	klog.Infof("init model manager successfully")

	return nil
}

// GetModelChannel gets model channel
func (mm *ModelManager) GetModelChannel(name string) chan Model {
	m, ok := mm.ModelChannelMap[name]
	if !ok {
		return nil
	}
	return m
}

// addNewModel adds model to the channel
func (mm *ModelManager) addNewModel(name string, model Model) {
	if _, ok := mm.ModelChannelMap[name]; !ok {
		mm.ModelChannelMap[name] = make(chan Model, ModelChannelCacheSize)
	}

	mm.ModelChannelMap[name] <- model
}

// handleMessage handles the message from GlobalManager
func (mm *ModelManager) handleMessage(message *wsclient.Message) {
	uniqueIdentifier := util.GetUniqueIdentifier(message.Header.Namespace, message.Header.ResourceName, message.Header.ResourceKind)
	switch message.Header.Operation {
	case InsertOperation:
		if err := mm.insertModel(uniqueIdentifier, message.Content); err != nil {
			klog.Errorf("insert %s(name=%s) to db failed, error: %v", message.Header.ResourceKind, uniqueIdentifier, err)
		}

	case DeleteOperation:
		if err := mm.deleteModel(uniqueIdentifier); err != nil {
			klog.Errorf("delete model(name=%s) in db failed, error: %v", uniqueIdentifier, err)
		}
	}
}

// insertModel inserts model config to db
func (mm *ModelManager) insertModel(name string, payload []byte) error {
	model := Model{}

	if err := json.Unmarshal(payload, &model); err != nil {
		return err
	}

	metaData, err := json.Marshal(model.MetaData)
	if err != nil {
		return err
	}

	spec, err := json.Marshal(model.Spec)
	if err != nil {
		return err
	}

	r := db.Resource{
		Name:       name,
		APIVersion: model.APIVersion,
		Kind:       model.Kind,
		MetaData:   string(metaData),
		Spec:       string(spec),
	}

	if err = db.SaveResource(&r); err != nil {
		return err
	}

	mm.addNewModel(name, model)

	return err
}

// deleteModel deletes model in db
func (mm *ModelManager) deleteModel(name string) error {
	if err := db.DeleteResource(name); err != nil {
		return err
	}

	if modelChannel := mm.ModelChannelMap[name]; modelChannel != nil {
		close(modelChannel)
		delete(mm.ModelChannelMap, name)
	}

	return nil
}
