package utils

import (
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"

	"github.com/kubeedge/sedna/pkg/globalmanager/config"
)

// KubeConfig from flags
func KubeConfig() (conf *rest.Config, err error) {
	kubeConfig, err := clientcmd.BuildConfigFromFlags(config.Config.Master,
		config.Config.KubeConfig)
	if err != nil {
		return nil, err
	}
	kubeConfig.ContentType = "application/json"

	return kubeConfig, nil
}
