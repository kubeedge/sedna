package v1alpha1

import (
	"ct_exporter/api/types/v1alpha1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
)

type ExampleV1Alpha1Interface interface {
	Federated(namespace string) FederatedInterface
	Dataset(namespace string) DatasetInterface
}

type ExampleV1Alpha1Client struct {
	restClient rest.Interface
}

func NewForConfig(c *rest.Config) (*ExampleV1Alpha1Client, error) {
	config := *c
	config.ContentConfig.GroupVersion = &schema.GroupVersion{Group: v1alpha1.GroupName, Version: v1alpha1.GroupVersion}
	config.APIPath = "/apis"
	config.NegotiatedSerializer = scheme.Codecs.WithoutConversion()
	config.UserAgent = rest.DefaultKubernetesUserAgent()

	client, err := rest.RESTClientFor(&config)
	if err != nil {
		return nil, err
	}

	return &ExampleV1Alpha1Client{restClient: client}, nil
}

func (c *ExampleV1Alpha1Client) FederatedClient(namespace string) FederatedInterface {
	return &FederatedClient{
		restClient: c.restClient,
		ns:         namespace,
	}
}

func (c *ExampleV1Alpha1Client) DatasetClient(namespace string) DatasetInterface {
	return &DatasetClient{
		restClient: c.restClient,
		ns:         namespace,
	}
}
