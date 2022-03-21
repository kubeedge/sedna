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
	"math/rand"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeinformers "k8s.io/client-go/informers"
	"k8s.io/klog/v2"

	clientset "github.com/kubeedge/sedna/pkg/client/clientset/versioned"
	sednainformers "github.com/kubeedge/sedna/pkg/client/informers/externalversions"
	"github.com/kubeedge/sedna/pkg/globalmanager/config"
	"github.com/kubeedge/sedna/pkg/globalmanager/messagelayer"
	websocket "github.com/kubeedge/sedna/pkg/globalmanager/messagelayer/ws"
	"github.com/kubeedge/sedna/pkg/globalmanager/runtime"
	"github.com/kubeedge/sedna/pkg/globalmanager/utils"
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

func genResyncPeriod(minPeriod time.Duration) time.Duration {
	factor := rand.Float64() + 1
	// [minPeriod, 2*minPeriod)
	return time.Duration(factor * float64(minPeriod.Nanoseconds()))
}

// Start starts the controllers it has managed
func (m *Manager) Start() error {
	kubeClient, err := utils.KubeClient()
	if err != nil {
		return err
	}

	kubecfg, err := utils.KubeConfig()
	if err != nil {
		return err
	}

	sednaClient, err := clientset.NewForConfig(kubecfg)
	if err != nil {
		return err
	}

	cfg := m.Config
	namespace := cfg.Namespace
	if namespace == "" {
		namespace = metav1.NamespaceAll
	}

	minResyncPeriod := time.Second * time.Duration(m.Config.MinResyncPeriodSeconds)

	kubeInformerFactory := kubeinformers.NewSharedInformerFactoryWithOptions(kubeClient, genResyncPeriod(minResyncPeriod), kubeinformers.WithNamespace(namespace))

	sednaInformerFactory := sednainformers.NewSharedInformerFactoryWithOptions(sednaClient, genResyncPeriod(minResyncPeriod), sednainformers.WithNamespace(namespace))

	context := &runtime.ControllerContext{
		Config: m.Config,

		KubeClient:          kubeClient,
		KubeInformerFactory: kubeInformerFactory,

		SednaClient:          sednaClient,
		SednaInformerFactory: sednaInformerFactory,
	}

	uc, _ := NewUpstreamController(context)

	downstreamSendFunc := messagelayer.NewContextMessageLayer().SendResourceObject

	stopCh := make(chan struct{})

	go uc.Run(stopCh)

	for name, factory := range NewRegistry() {
		f, err := factory(context)
		if err != nil {
			return fmt.Errorf("failed to initialize controller %s: %v", name, err)
		}
		f.SetDownstreamSendFunc(downstreamSendFunc)
		f.SetUpstreamHandler(uc.Add)

		klog.Infof("initialized controller %s", name)
		go f.Run(stopCh)
	}

	kubeInformerFactory.Start(stopCh)
	sednaInformerFactory.Start(stopCh)

	addr := fmt.Sprintf("%s:%d", m.Config.WebSocket.Address, m.Config.WebSocket.Port)

	ws := websocket.NewServer(addr)
	err = ws.ListenAndServe()
	if err != nil {
		close(stopCh)
		return fmt.Errorf("failed to listen websocket at %s: %v", addr, err)
	}
	return nil
}
