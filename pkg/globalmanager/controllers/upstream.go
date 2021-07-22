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

package controllers

import (
	"fmt"
	"strings"

	"github.com/kubeedge/sedna/pkg/globalmanager/config"
	"github.com/kubeedge/sedna/pkg/globalmanager/messagelayer"
	"github.com/kubeedge/sedna/pkg/globalmanager/runtime"
	"k8s.io/klog/v2"
)

// UpstreamController subscribes the updates from edge and syncs to k8s api server
type UpstreamController struct {
	messageLayer   messagelayer.MessageLayer
	updateHandlers map[string]runtime.UpstreamUpdateHandler
}

func (uc *UpstreamController)checkOperation(operation string) error {
	// current only support the 'status' operation
	if operation != "status" {
		return fmt.Errorf("unknown operation %s", operation)
	}
	return nil
}

// syncEdgeUpdate receives the updates from edge and syncs these to k8s.
func (uc *UpstreamController) syncEdgeUpdate() {
	for {
		select {
		case <-uc.messageLayer.Done():
			klog.Info("Stop sedna upstream loop")
			return
		default:
		}

		update, err := uc.messageLayer.ReceiveResourceUpdate()
		if err == nil {
			err = uc.checkOperation(update.operation)
		}
		if err != nil && err :=  {
			klog.Warningf("Ignore update since this err: %+v", err)
			continue
		}
		if err != nil 

		kind := update.Kind
		namespace := update.Namespace
		name := update.Name
		operation := update.Operation

		handler, ok := uc.updateHandlers[kind]
		if ok {
			err := handler(name, namespace, operation, update.Content)
			if err != nil {
				klog.Errorf("Error to handle %s %s/%s operation(%s): %+v", kind, namespace, name, operation, err)
			}
		} else {
			klog.Warningf("No handler for resource kind %s", kind)
		}
	}
}

// Start the upstream controller
func (uc *UpstreamController) Start() error {
	klog.Info("Start the sedna upstream controller")

	go uc.syncEdgeUpdate()
	return nil
}

// GetName returns the name of the upstream controller
func (uc *UpstreamController) GetName() string {
	return "UpstreamController"
}

func (uc *UpstreamController) Add(kind string, handler runtime.UpstreamUpdateHandler) error {
	kind = strings.ToLower(kind)
	if _, ok := uc.updateHandlers[kind]; ok {
		return fmt.Errorf("a upstream handler for kind %s already exists", kind)
	}
	uc.updateHandlers[kind] = handler

	return nil
}

// NewUpstreamController creates a new Upstream controller from config
func NewUpstreamController(cfg *config.ControllerConfig) (*UpstreamController, error) {
	uc := &UpstreamController{
		messageLayer:   messagelayer.NewContextMessageLayer(),
		updateHandlers: make(map[string]runtime.UpstreamUpdateHandler),
	}

	return uc, nil
}
