package dataset

import (
	"context"
	"encoding/json"

	sednav1 "github.com/kubeedge/sedna/pkg/apis/sedna/v1alpha1"
	"github.com/kubeedge/sedna/pkg/globalmanager/config"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/cache"

	sednaclientset "github.com/kubeedge/sedna/pkg/client/clientset/versioned/typed/sedna/v1alpha1"
	sednav1listers "github.com/kubeedge/sedna/pkg/client/listers/sedna/v1alpha1"
	"github.com/kubeedge/sedna/pkg/globalmanager/runtime"
)

// Controller handles all dataset objects including: syncing to edge and update from edge.
type Controller struct {
	client sednaclientset.SednaV1alpha1Interface

	storeSynced cache.InformerSynced

	// A store of dataset
	lister sednav1listers.DatasetLister

	cfg *config.ControllerConfig
}

// updateDatasetFromEdge syncs update from edge
func (c *Controller) updateDatasetFromEdge(name, namespace, operation string, content []byte) error {
	status := sednav1.DatasetStatus{}
	err := json.Unmarshal(content, &status)
	if err != nil {
		return err
	}

	return c.updateDatasetStatus(name, namespace, status)
}

// updateDatasetStatus updates the dataset status
func (c *Controller) updateDatasetStatus(name, namespace string, status sednav1.DatasetStatus) error {
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
