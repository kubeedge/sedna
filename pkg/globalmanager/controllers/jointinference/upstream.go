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

package jointinference

import (
	"context"
	"encoding/json"
	"fmt"

	sednav1 "github.com/kubeedge/sedna/pkg/apis/sedna/v1alpha1"
	"github.com/kubeedge/sedna/pkg/globalmanager/runtime"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"
)

func (c *Controller) updateMetrics(name, namespace string, metrics []sednav1.Metric) error {
	client := c.client.JointInferenceServices(namespace)

	return runtime.RetryUpdateStatus(name, namespace, func() error {
		joint, err := client.Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		joint.Status.Metrics = metrics
		_, err = client.UpdateStatus(context.TODO(), joint, metav1.UpdateOptions{})
		return err
	})
}

// updateFromEdge syncs the edge updates to k8s
func (c *Controller) updateFromEdge(name, namespace, operation string, content []byte) error {
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

	err := json.Unmarshal(content, &status)
	if err != nil {
		return err
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

	metrics := runtime.ConvertMapToMetrics(info)

	err = c.updateMetrics(name, namespace, metrics)
	if err != nil {
		return fmt.Errorf("failed to update metrics, err:%w", err)
	}
	return nil
}

func (c *Controller) SetUpstreamHandler(addFunc runtime.UpstreamHandlerAddFunc) error {
	return addFunc(KindName, c.updateFromEdge)
}
