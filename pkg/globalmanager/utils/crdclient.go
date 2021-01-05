package utils

import (
	"k8s.io/klog/v2"

	clientset "github.com/edgeai-neptune/neptune/pkg/client/clientset/versioned/typed/neptune/v1alpha1"
)

// NewCRDClient is used to create a restClient for crd
func NewCRDClient() (*clientset.NeptuneV1alpha1Client, error) {
	cfg, _ := KubeConfig()
	client, err := clientset.NewForConfig(cfg)
	if err != nil {
		klog.Errorf("Failed to create REST Client due to error %v", err)
		return nil, err
	}

	return client, nil
}
