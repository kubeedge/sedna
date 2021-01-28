package utils

import (
	"k8s.io/klog/v2"

	clientset "github.com/kubeedge/sedna/pkg/client/clientset/versioned/typed/sedna/v1alpha1"
)

// NewCRDClient is used to create a restClient for crd
func NewCRDClient() (*clientset.SednaV1alpha1Client, error) {
	cfg, _ := KubeConfig()
	client, err := clientset.NewForConfig(cfg)
	if err != nil {
		klog.Errorf("Failed to create REST Client due to error %v", err)
		return nil, err
	}

	return client, nil
}
