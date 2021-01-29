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

	sednav1 "github.com/kubeedge/sedna/pkg/apis/sedna/v1alpha1"
	"github.com/kubeedge/sedna/pkg/localcontroller/db"
	"github.com/kubeedge/sedna/pkg/localcontroller/gmclient"
	"github.com/kubeedge/sedna/pkg/localcontroller/util"
)

// ModelManager defines model manager
type ModelManager struct {
	Client   gmclient.ClientI
	ModelMap map[string]sednav1.Model
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
		ModelMap: make(map[string]sednav1.Model),
		Client:   client,
	}

	return &mm
}

// Start starts model manager
func (mm *ModelManager) Start() error {
	return nil
}

// GetModel gets model
func (mm *ModelManager) GetModel(name string) (sednav1.Model, bool) {
	model, ok := mm.ModelMap[name]
	return model, ok
}

// addNewModel adds model
func (mm *ModelManager) addNewModel(name string, model sednav1.Model) {
	mm.ModelMap[name] = model
}

// insertModel inserts model config to db
func (mm *ModelManager) Insert(message *gmclient.Message) error {
	model := sednav1.Model{}
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
