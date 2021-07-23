package dataset

import (
	"context"
	"encoding/json"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	sednav1 "github.com/kubeedge/sedna/pkg/apis/sedna/v1alpha1"
	"github.com/kubeedge/sedna/pkg/globalmanager/config"
	"github.com/kubeedge/sedna/pkg/globalmanager/runtime"

	sednaclientset "github.com/kubeedge/sedna/pkg/client/clientset/versioned/typed/sedna/v1alpha1"
)

const (
	// KindName is the kind name of CR this controller controls
	KindName = "Dataset"

	// Name is this controller name
	Name = "Dataset"
)

// Controller handles all dataset objects including: syncing to edge and update from edge.
type Controller struct {
	client sednaclientset.SednaV1alpha1Interface

	cfg *config.ControllerConfig
}

// updateFromEdge syncs update from edge
func (c *Controller) updateFromEdge(name, namespace, operation string, content []byte) error {
	status := sednav1.DatasetStatus{}
	err := json.Unmarshal(content, &status)
	if err != nil {
		return err
	}

	return c.updateStatus(name, namespace, status)
}

// updateStatus updates the dataset status
func (c *Controller) updateStatus(name, namespace string, status sednav1.DatasetStatus) error {
	client := c.client.Datasets(namespace)

	if status.UpdateTime == nil {
		now := metav1.Now()
		status.UpdateTime = &now
	}

	return runtime.RetryUpdateStatus(name, namespace, func() error {
		dataset, err := client.Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		dataset.Status = status
		_, err = client.UpdateStatus(context.TODO(), dataset, metav1.UpdateOptions{})
		return err
	})
}

func (c *Controller) Run(stopCh <-chan struct{}) {
	// noop now
}

// New creates a dataset controller
func New(cc *runtime.ControllerContext) (runtime.FeatureControllerI, error) {
	c := &Controller{
		client: cc.SednaClient.SednaV1alpha1(),
	}

	// only upstream
	cc.UpstreamController.Add(KindName, c.updateFromEdge)

	return c, nil
}
