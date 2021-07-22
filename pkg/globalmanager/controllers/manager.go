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

	"k8s.io/klog/v2"

	"github.com/kubeedge/sedna/pkg/globalmanager/config"
	websocket "github.com/kubeedge/sedna/pkg/globalmanager/messagelayer/ws"
	"github.com/kubeedge/sedna/pkg/globalmanager/runtime"
)

// Manager defines the controller manager
type Manager struct {
	Config *config.ControllerConfig
}

// New creates the controller manager
func New(cc *config.ControllerConfig) *Manager {
	config.InitConfigure(cc)
	return &Manager{
		Config: cc,
	}
}

// Start starts the controllers it has managed
func (m *Manager) Start() error {
	uc, _ := NewUpstreamController(m.Config)
	dc, _ := NewDownstreamController(m.Config)
	uc.Start()
	dc.Start()
	context := &runtime.ControllerContext{
		UpstreamController: uc,
		Config:             m.Config,
	}

	for name, factory := range NewRegistry() {
		f, err := factory(context)
		if err != nil {
			return fmt.Errorf("failed to initialize controller %s: %v", name, err)
		}
		err = f.Start()
		if err != nil {
			return fmt.Errorf("failed to start controller %s: %v", name, err)
		}
		klog.Infof("started controller %s", name)
	}

	addr := fmt.Sprintf("%s:%d", m.Config.WebSocket.Address, m.Config.WebSocket.Port)

	ws := websocket.NewServer(addr)
	err := ws.ListenAndServe()
	if err != nil {
		return fmt.Errorf("failed to listen websocket at %s: %v", addr, err)
	}
	return nil
}
