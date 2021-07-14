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

package globalmanager

import (
	"fmt"
	"os"

	"k8s.io/klog/v2"

	"github.com/kubeedge/sedna/pkg/globalmanager/config"
	websocket "github.com/kubeedge/sedna/pkg/globalmanager/messagelayer/ws"
)

// MainController defines the main controller
type MainController struct {
	Config *config.ControllerConfig
}

// NewController creates a new main controller
func NewController(cc *config.ControllerConfig) *MainController {
	config.InitConfigure(cc)
	return &MainController{
		Config: cc,
	}
}

// Start starts the main controller
func (c *MainController) Start() {
	type newFunc func(cfg *config.ControllerConfig) (FeatureControllerI, error)

	for _, featureFunc := range []newFunc{
		NewUpstreamController,
		NewDownstreamController,
		NewFederatedController,
		NewJointController,
		NewIncrementalJobController,
		NewLifelongLearningJobController,
		NewMultiEdgeTrackingController,
	} {
		f, _ := featureFunc(c.Config)
		err := f.Start()
		if err != nil {
			klog.Warningf("failed to start controller %s: %+v", f.GetName(), err)
		} else {
			klog.Infof("started controller %s", f.GetName())
		}
	}

	addr := fmt.Sprintf("%s:%d", c.Config.WebSocket.Address, c.Config.WebSocket.Port)

	ws := websocket.NewServer(addr)
	err := ws.ListenAndServe()
	if err != nil {
		klog.Fatalf("failed to listen websocket at %s", addr)
		os.Exit(1)
	}
}
