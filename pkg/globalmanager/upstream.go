package globalmanager

import (
	"context"
	"encoding/json"
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"

	neptunev1 "github.com/edgeai-neptune/neptune/pkg/apis/neptune/v1alpha1"
	clientset "github.com/edgeai-neptune/neptune/pkg/client/clientset/versioned/typed/neptune/v1alpha1"
	"github.com/edgeai-neptune/neptune/pkg/globalmanager/config"
	"github.com/edgeai-neptune/neptune/pkg/globalmanager/messagelayer"
	"github.com/edgeai-neptune/neptune/pkg/globalmanager/utils"
)

// updateHandler handles the updates from LC(running at edge) to update the
// corresponding resource
type updateHandler func(namespace, name, operation string, content []byte) error

// UpstreamController subscribes the updates from edge and syncs to k8s api server
type UpstreamController struct {
	client         *clientset.NeptuneV1alpha1Client
	messageLayer   messagelayer.MessageLayer
	updateHandlers map[string]updateHandler
}

const upstreamStatusUpdateRetries = 3

// retryUpdateStatus simply retries to call the status update func
func retryUpdateStatus(name, namespace string, updateStatusFunc func() error) error {
	var err error
	for retry := 0; retry <= upstreamStatusUpdateRetries; retry++ {
		err = updateStatusFunc()
		if err == nil {
			return nil
		}
		klog.Warningf("Error to update %s/%s status, retried %d times: %+v", namespace, name, retry, err)
	}
	return err
}

func newUnmarshalError(namespace, name, operation string, content []byte) error {
	return fmt.Errorf("Unable to unmarshal content for (%s/%s) operation: '%s', content: '%+v'", namespace, name, operation, string(content))
}

// updateDatasetStatus updates the dataset status
func (uc *UpstreamController) updateDatasetStatus(name, namespace string, status neptunev1.DatasetStatus) error {
	client := uc.client.Datasets(namespace)

	if status.UpdateTime == nil {
		now := metav1.Now()
		status.UpdateTime = &now
	}

	return retryUpdateStatus(name, namespace, func() error {
		dataset, err := client.Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		dataset.Status = status
		_, err = client.UpdateStatus(context.TODO(), dataset, metav1.UpdateOptions{})
		return err
	})
}

// updateDatasetFromEdge syncs update from edge
func (uc *UpstreamController) updateDatasetFromEdge(name, namespace, operation string, content []byte) error {
	if operation != "status" {
		return fmt.Errorf("unknown operation %s", operation)
	}

	status := neptunev1.DatasetStatus{}
	err := json.Unmarshal(content, &status)
	if err != nil {
		return newUnmarshalError(namespace, name, operation, content)
	}

	return uc.updateDatasetStatus(name, namespace, status)
}

// convertToMetrics converts the metrics from LCs to resource metrics
func convertToMetrics(m map[string]interface{}) []neptunev1.Metric {
	var l []neptunev1.Metric
	for k, v := range m {
		var displayValue string
		switch t := v.(type) {
		case string:
			displayValue = t
		default:
			// ignore the json marshal error
			b, _ := json.Marshal(v)
			displayValue = string(b)
		}

		l = append(l, neptunev1.Metric{Key: k, Value: displayValue})
	}
	return l
}

func (uc *UpstreamController) updateJointInferenceMetrics(name, namespace string, metrics []neptunev1.Metric) error {
	client := uc.client.JointInferenceServices(namespace)

	return retryUpdateStatus(name, namespace, func() error {
		joint, err := client.Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		joint.Status.Metrics = metrics
		_, err = client.UpdateStatus(context.TODO(), joint, metav1.UpdateOptions{})
		return err
	})
}

// updateJointInferenceFromEdge syncs the edge updates to k8s
func (uc *UpstreamController) updateJointInferenceFromEdge(name, namespace, operation string, content []byte) error {
	// current only support the 'status' operation
	if operation != "status" {
		return fmt.Errorf("unknown operation %s", operation)
	}

	var err error

	// Output defines owner output information
	type Output struct {
		ServiceInfo map[string]interface{} `json:"ownerInfo"`
	}

	var status struct {
		// Phase always should be "inference"
		Phase  string  `json:"phase"`
		Status string  `json:"status"`
		Output *Output `json:"output"`
	}

	err = json.Unmarshal(content, &status)
	if err != nil {
		return newUnmarshalError(namespace, name, operation, content)
	}

	// TODO: propagate status.Status to k8s

	output := status.Output
	if output == nil || output.ServiceInfo == nil {
		// no output info
		klog.Warningf("empty status info for joint inference service %s/%s", namespace, name)
		return nil
	}

	info := output.ServiceInfo

	for _, ignoreTimeKey := range []string{
		"startTime",
		"updateTime",
	} {
		delete(info, ignoreTimeKey)
	}

	metrics := convertToMetrics(info)

	err = uc.updateJointInferenceMetrics(name, namespace, metrics)
	if err != nil {
		return fmt.Errorf("failed to update metrics, err:%+w", err)
	}
	return nil
}

// syncEdgeUpdate receives the updates from edge and syncs these to k8s.
func (uc *UpstreamController) syncEdgeUpdate() {
	for {
		select {
		case <-uc.messageLayer.Done():
			klog.Info("Stop neptune upstream loop")
			return
		default:
		}

		update, err := uc.messageLayer.ReceiveResourceUpdate()
		if err != nil {
			klog.Warningf("Ignore update since this err: %+v", err)
			continue
		}

		kind := update.Kind
		namespace := update.Namespace
		name := update.Name
		operation := update.Operation

		handler, ok := uc.updateHandlers[kind]
		if ok {
			err := handler(name, namespace, operation, update.Content)
			if err != nil {
				klog.Errorf("Error to handle %s %s/%s operation(%s): %+v", kind, namespace, name, operation, err)
			}
		} else {
			klog.Warningf("No handler for resource kind %s", kind)
		}
	}
}

// Start the upstream controller
func (uc *UpstreamController) Start() error {
	klog.Info("Start the neptune upstream controller")

	go uc.syncEdgeUpdate()
	return nil
}

func (uc *UpstreamController) GetName() string {
	return "UpstreamController"
}

// NewUpstreamController create UpstreamController from config
func NewUpstreamController(cfg *config.ControllerConfig) (FeatureControllerI, error) {
	client, err := utils.NewCRDClient()
	if err != nil {
		return nil, fmt.Errorf("create crd client failed with error: %w", err)
	}
	uc := &UpstreamController{
		client:       client,
		messageLayer: messagelayer.NewContextMessageLayer(),
	}

	// NOTE: current no direct model update from edge,
	// model update will be triggered by the corresponding training feature
	uc.updateHandlers = map[string]updateHandler{
		"dataset":               uc.updateDatasetFromEdge,
		"jointinferenceservice": uc.updateJointInferenceFromEdge,
	}

	return uc, nil
}
