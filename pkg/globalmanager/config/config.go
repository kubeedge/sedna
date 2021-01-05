package config

import (
	"io/ioutil"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/klog/v2"
	"sigs.k8s.io/yaml"

	"github.com/edgeai-neptune/neptune/pkg/util"
)

const (
	DefaultKubeConfig       = ""
	DefaultNamespace        = v1.NamespaceAll
	DefaultWebsocketAddress = "0.0.0.0"
	DefaultWebsocketPort    = 9000
	DefaultLCServer         = "http://localhost:9100"
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
	// ImageHub indicates the image which the framework/version mapping to
	// +Required
	ImageHub map[string]string `json:"imageHub,omitempty"`

	// websocket server config
	// Since the current limit of kubeedge(1.5), GM needs to build the websocket channel for communicating between GM and LCs.
	WebSocket WebSocket `json:"websocket,omitempty"`

	// lc config to info the worker
	LC LCConfig `json:"localController,omitempty"`
}

type WebSocket struct {
	// default DefaultWebsocketAddress
	Address string `json:"address,omitempty"`
	// default DefaultWebsocketPort
	Port int64 `json:"port,omitempty"`
}

type LCConfig struct {
	// default DefaultLCServer
	Server string `json:"server"`
}

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

func (c *ControllerConfig) Validate() field.ErrorList {
	allErrs := field.ErrorList{}
	if c.KubeConfig != "" && !util.FileIsExist(c.KubeConfig) {
		allErrs = append(allErrs, field.Invalid(field.NewPath("kubeconfig"), c.KubeConfig, "kubeconfig not exist"))
	}
	return allErrs
}

func NewDefaultControllerConfig() *ControllerConfig {
	return &ControllerConfig{
		KubeConfig: DefaultKubeConfig,
		Master:     "",
		Namespace:  DefaultNamespace,
		WebSocket: WebSocket{
			Address: DefaultWebsocketAddress,
			Port:    DefaultWebsocketPort,
		},
		LC: LCConfig{
			Server: DefaultLCServer,
		},
	}
}

var Config ControllerConfig

func InitConfigure(cc *ControllerConfig) {
	Config = *cc
}
