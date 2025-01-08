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

package model

import (
	"encoding/json"

	sednav1 "github.com/kubeedge/sedna/pkg/apis/sedna/v1alpha1"
	"github.com/kubeedge/sedna/pkg/localcontroller/db"
	clienttypes "github.com/kubeedge/sedna/pkg/localcontroller/gmclient"
	"github.com/kubeedge/sedna/pkg/localcontroller/util"
	workertypes "github.com/kubeedge/sedna/pkg/localcontroller/worker"
)

// ModelManager defines model manager
type Manager struct {
	Client   clienttypes.ClientI
	ModelMap map[string]sednav1.Model
}

const (
	// KindName is kind of model resource
	KindName = "model"
)

// New creates a model manager
func New(client clienttypes.ClientI) *Manager {
	mm := Manager{
		ModelMap: make(map[string]sednav1.Model),
		Client:   client,
	}

	return &mm
}

// Start starts model manager
func (mm *Manager) Start() error {
	return nil
}

// GetModel gets model
func (mm *Manager) GetModel(name string) (sednav1.Model, bool) {
	model, ok := mm.ModelMap[name]
	return model, ok
}

// addNewModel adds model
func (mm *Manager) addNewModel(name string, model sednav1.Model) {
	mm.ModelMap[name] = model
}

// insertModel inserts model config to db
func (mm *Manager) Insert(message *clienttypes.Message) error {
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
func (mm *Manager) Delete(message *clienttypes.Message) error {
	name := util.GetUniqueIdentifier(message.Header.Namespace, message.Header.ResourceName, message.Header.ResourceKind)

	delete(mm.ModelMap, name)

	if err := db.DeleteResource(name); err != nil {
		return err
	}

	return nil
}

func (mm *Manager) GetName() string {
	return KindName
}

func (mm *Manager) AddWorkerMessage(_ workertypes.MessageContent) {
	// dummy
}
