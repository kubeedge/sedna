package manager

import (
	"encoding/json"

	neptunev1 "github.com/edgeai-neptune/neptune/pkg/apis/neptune/v1alpha1"
	"github.com/edgeai-neptune/neptune/pkg/localcontroller/db"
	"github.com/edgeai-neptune/neptune/pkg/localcontroller/gmclient"
	"github.com/edgeai-neptune/neptune/pkg/localcontroller/util"
)

// ModelManager defines model manager
type ModelManager struct {
	Client   gmclient.ClientI
	ModelMap map[string]neptunev1.Model
}

const (
	// ModelCacheSize is size of cache
	ModelCacheSize = 100
	// ModelResourceKind is kind of dataset resource
	ModelResourceKind = "model"
)

// NewModelManager creates a model manager
func NewModelManager(client gmclient.ClientI) *ModelManager {
	mm := ModelManager{
		ModelMap: make(map[string]neptunev1.Model),
		Client:   client,
	}

	return &mm
}

// Start starts model manager
func (mm *ModelManager) Start() error {
	return nil
}

// GetModel gets model
func (mm *ModelManager) GetModel(name string) (neptunev1.Model, bool) {
	model, ok := mm.ModelMap[name]
	return model, ok
}

// addNewModel adds model
func (mm *ModelManager) addNewModel(name string, model neptunev1.Model) {
	mm.ModelMap[name] = model
}

// insertModel inserts model config to db
func (mm *ModelManager) Insert(message *gmclient.Message) error {
	model := neptunev1.Model{}
	name := util.GetUniqueIdentifier(message.Header.Namespace, message.Header.ResourceName, message.Header.ResourceKind)

	if err := json.Unmarshal(message.Content, &model); err != nil {
		return err
	}

	if err := db.SaveResource(name, model.TypeMeta, model.ObjectMeta, model.Spec); err != nil {
		return err
	}

	mm.addNewModel(name, model)

	return nil
}

// Delete deletes model in db
func (mm *ModelManager) Delete(message *gmclient.Message) error {
	name := util.GetUniqueIdentifier(message.Header.Namespace, message.Header.ResourceName, message.Header.ResourceKind)

	delete(mm.ModelMap, name)

	if err := db.DeleteResource(name); err != nil {
		return err
	}

	return nil
}

func (mm *ModelManager) GetName() string {
	return ModelResourceKind
}

func (mm *ModelManager) AddWorkerMessage(message WorkerMessage) {
	// dummy
}
