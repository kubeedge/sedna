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

package featureextraction

import (
	"context"
	"encoding/json"
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"

	sednav1 "github.com/kubeedge/sedna/pkg/apis/sedna/v1alpha1"
	"github.com/kubeedge/sedna/pkg/globalmanager/runtime"
)

// updateHandler handles the updates from LC(running at edge) to update the
// corresponding resource
type updateHandler func(namespace, name, operation string, content []byte) error

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

func checkUpstreamOperation(operation string) error {
	// current only support the 'status' operation
	if operation != "status" {
		return fmt.Errorf("unknown operation %s", operation)
	}
	return nil
}

// convertToMetrics converts the metrics from LCs to resource metrics
func convertToMetrics(m map[string]interface{}) []sednav1.Metric {
	var l []sednav1.Metric
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

		l = append(l, sednav1.Metric{Key: k, Value: displayValue})
	}
	return l
}

func (c *Controller) updateFeatureExtractionMetrics(name, namespace string, metrics []sednav1.Metric) error {
	client := c.client.FeatureExtractionServices(namespace)

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

func (c *Controller) updateFeatureExtractionFromEdge(name, namespace, operation string, content []byte) error {
	err := checkUpstreamOperation(operation)
	if err != nil {
		return err
	}

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
		klog.Warningf("empty status info for feature extraction service %s/%s", namespace, name)
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

	err = c.updateFeatureExtractionMetrics(name, namespace, metrics)
	if err != nil {
		return fmt.Errorf("failed to update metrics, err:%+w", err)
	}
	return nil
}

func (c *Controller) SetUpstreamHandler(addFunc runtime.UpstreamHandlerAddFunc) error {
	return addFunc(KindName, c.updateFeatureExtractionFromEdge)
}
