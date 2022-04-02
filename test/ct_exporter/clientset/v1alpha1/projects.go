package v1alpha1

import (
	"context"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	"ct_exporter/api/types/v1alpha1"
)

type FederatedInterface interface {
	List(ctx context.Context, opts metav1.ListOptions) (*v1alpha1.FederatedLearningJobList, error)
	Get(ctx context.Context, name string, options metav1.GetOptions) (*v1alpha1.FederatedLearningJob, error)
	Create(ctx context.Context, project *v1alpha1.FederatedLearningJob) (*v1alpha1.FederatedLearningJob, error)
	Watch(ctx context.Context, opts metav1.ListOptions) (watch.Interface, error)
	// ...
}

type DatasetInterface interface {
	List(ctx context.Context, opts metav1.ListOptions) (*v1alpha1.DatasetList, error)
	Get(ctx context.Context, name string, options metav1.GetOptions) (*v1alpha1.Dataset, error)
	Create(ctx context.Context, project *v1alpha1.Dataset) (*v1alpha1.Dataset, error)
	Watch(ctx context.Context, opts metav1.ListOptions) (watch.Interface, error)
	// ...
}

type FederatedClient struct {
	restClient rest.Interface
	ns         string
}

type DatasetClient struct {
	restClient rest.Interface
	ns         string
}

func (c *FederatedClient) List(ctx context.Context, opts metav1.ListOptions) (*v1alpha1.FederatedLearningJobList, error) {
	result := v1alpha1.FederatedLearningJobList{}
	err := c.restClient.
		Get().
		Namespace(c.ns).
		Resource("federatedlearningjobs").
		VersionedParams(&opts, scheme.ParameterCodec).
		Do(ctx).
		Into(&result)

	return &result, err
}

func (c *FederatedClient) Get(ctx context.Context, name string, opts metav1.GetOptions) (*v1alpha1.FederatedLearningJob, error) {
	result := v1alpha1.FederatedLearningJob{}
	err := c.restClient.
		Get().
		Namespace(c.ns).
		Resource("federatedlearningjobs").
		Name(name).
		VersionedParams(&opts, scheme.ParameterCodec).
		Do(ctx).
		Into(&result)

	return &result, err
}

func (c *FederatedClient) Create(ctx context.Context, project *v1alpha1.FederatedLearningJob) (*v1alpha1.FederatedLearningJob, error) {
	result := v1alpha1.FederatedLearningJob{}
	err := c.restClient.
		Post().
		Namespace(c.ns).
		Resource("federatedlearningjobs").
		Body(project).
		Do(ctx).
		Into(&result)

	return &result, err
}

func (c *FederatedClient) Watch(ctx context.Context, opts metav1.ListOptions) (watch.Interface, error) {
	opts.Watch = true
	return c.restClient.
		Get().
		Namespace(c.ns).
		Resource("federatedllearningjobs").
		VersionedParams(&opts, scheme.ParameterCodec).
		Watch(ctx)
}

func (c *DatasetClient) List(ctx context.Context, opts metav1.ListOptions) (*v1alpha1.DatasetList, error) {
	result := v1alpha1.DatasetList{}
	err := c.restClient.
		Get().
		Namespace(c.ns).
		Resource("datasets").
		VersionedParams(&opts, scheme.ParameterCodec).
		Do(ctx).
		Into(&result)

	return &result, err
}

func (c *DatasetClient) Get(ctx context.Context, name string, opts metav1.GetOptions) (*v1alpha1.Dataset, error) {
	result := v1alpha1.Dataset{}
	err := c.restClient.
		Get().
		Namespace(c.ns).
		Resource("datasets").
		Name(name).
		VersionedParams(&opts, scheme.ParameterCodec).
		Do(ctx).
		Into(&result)

	return &result, err
}

func (c *DatasetClient) Create(ctx context.Context, project *v1alpha1.Dataset) (*v1alpha1.Dataset, error) {
	result := v1alpha1.Dataset{}
	err := c.restClient.
		Post().
		Namespace(c.ns).
		Resource("datasets").
		Body(project).
		Do(ctx).
		Into(&result)

	return &result, err
}

func (c *DatasetClient) Watch(ctx context.Context, opts metav1.ListOptions) (watch.Interface, error) {
	opts.Watch = true
	return c.restClient.
		Get().
		Namespace(c.ns).
		Resource("datasets").
		VersionedParams(&opts, scheme.ParameterCodec).
		Watch(ctx)
}

