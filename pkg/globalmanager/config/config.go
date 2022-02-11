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

package config

import (
	"io/ioutil"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/klog/v2"
	"sigs.k8s.io/yaml"

	"github.com/kubeedge/sedna/pkg/util"
)

const (
	defaultKubeConfig       = ""
	defaultNamespace        = v1.NamespaceAll
	defaultWebsocketAddress = "0.0.0.0"
	defaultWebsocketPort    = 9000
	defaultLCServer         = "http://localhost:9100"
	defaultKBServer         = "http://localhost:9020"
	defaultPeriod           = 30
)

// ControllerConfig indicates the config of controller
type ControllerConfig struct {
	// KubeAPIConfig indicates the kubernetes cluster info which controller will connected
	KubeConfig string `json:"kubeConfig,omitempty"`

	// Master indicates the address of the Kubernetes API server. Overrides any value in KubeConfig.
	// such as https://127.0.0.1:8443
	// default ""
	Master string `json:"master"`
	// Namespace indicates which namespace the controller listening to.
	// default ""
	Namespace string `json:"namespace,omitempty"`

	// websocket server config
	// Since the current limit of kubeedge(1.5), GM needs to build the websocket channel for communicating between GM and LCs.
	WebSocket WebSocket `json:"websocket,omitempty"`

	// lc config to info the worker
	LC LCConfig `json:"localController,omitempty"`

	// kb config to info the worker
	KB KBConfig `json:"knowledgeBaseServer,omitempty"`

	// period config min resync period
	// default 30s
	MinResyncPeriodSeconds int64 `json:"minResyncPeriodSeconds,omitempty"`
}

// WebSocket describes GM of websocket config
type WebSocket struct {
	// default defaultWebsocketAddress
	Address string `json:"address,omitempty"`
	// default defaultWebsocketPort
	Port int64 `json:"port,omitempty"`
}

// LCConfig describes LC config to inject the worker
type LCConfig struct {
	// default defaultLCServer
	Server string `json:"server"`
}

// KBConfig describes KB config to inject the worker
type KBConfig struct {
	// default defaultKBServer
	Server string `json:"server"`
}

// Parse parses from filename
func (c *ControllerConfig) Parse(filename string) error {
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		klog.Errorf("Failed to read configfile %s: %v", filename, err)
		return err
	}
	err = yaml.Unmarshal(data, c)
	if err != nil {
		klog.Errorf("Failed to unmarshal configfile %s: %v", filename, err)
		return err
	}
	return nil
}

// Validate validate the config
func (c *ControllerConfig) Validate() field.ErrorList {
	allErrs := field.ErrorList{}
	if c.KubeConfig != "" && !util.FileIsExist(c.KubeConfig) {
		allErrs = append(allErrs, field.Invalid(field.NewPath("kubeconfig"), c.KubeConfig, "kubeconfig not exist"))
	}
	return allErrs
}

// NewDefaultControllerConfig creates default config
func NewDefaultControllerConfig() *ControllerConfig {
	return &ControllerConfig{
		KubeConfig: defaultKubeConfig,
		Master:     "",
		Namespace:  defaultNamespace,
		WebSocket: WebSocket{
			Address: defaultWebsocketAddress,
			Port:    defaultWebsocketPort,
		},
		LC: LCConfig{
			Server: defaultLCServer,
		},
		KB: KBConfig{
			Server: defaultKBServer,
		},
		MinResyncPeriodSeconds: defaultPeriod,
	}
}

// Config singleton for GM
var Config ControllerConfig

// InitConfigure inits config
func InitConfigure(cc *ControllerConfig) {
	Config = *cc
}
