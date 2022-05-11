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

package videoanalytics

import (
	"context"
	"encoding/json"
	"fmt"

	sednav1 "github.com/kubeedge/sedna/pkg/apis/sedna/v1alpha1"
	"github.com/kubeedge/sedna/pkg/globalmanager/runtime"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"
)

const upstreamStatusUpdateRetries = 3

func (c *Controller) updateModelMetrics(jobName, namespace string, metrics []sednav1.Metric) error {
	var err error
	job, err := c.client.VideoAnalyticsJobs(namespace).Get(context.TODO(), jobName, metav1.GetOptions{})
	if err != nil {
		// videoanalytics crd not found
		return err
	}
	modelName := job.Spec.Model.Name
	client := c.client.Models(namespace)

	return runtime.RetryUpdateStatus(modelName, namespace, (func() error {
		model, err := client.Get(context.TODO(), modelName, metav1.GetOptions{})
		if err != nil {
			return err
		}

		now := metav1.Now()
		model.Status.UpdateTime = &now
		model.Status.Metrics = metrics
		_, err = client.UpdateStatus(context.TODO(), model, metav1.UpdateOptions{})
		return err
	}))
}

func (c *Controller) appendStatusCondition(name, namespace string, cond sednav1.VideoAnalyticsJobCondition) error {
	client := c.client.VideoAnalyticsJobs(namespace)

	return runtime.RetryUpdateStatus(name, namespace, (func() error {
		job, err := client.Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		job.Status.Conditions = append(job.Status.Conditions, cond)
		_, err = client.UpdateStatus(context.TODO(), job, metav1.UpdateOptions{})
		return err
	}))
}

func newUnmarshalError(namespace, name, operation string, content []byte) error {
	return fmt.Errorf("Unable to unmarshal content for (%s/%s) operation: '%s', content: '%+v'", namespace, name, operation, string(content))
}

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

// updateFromEdge updates the videoanalytics job's status
func (c *Controller) updateFromEdge(name, namespace, operation string, content []byte) (err error) {
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

	output := status.Output
	if output == nil || output.ServiceInfo == nil {
		// no output info
		klog.Warningf("empty status info for videoanalytics job %s/%s", namespace, name)
		return nil
	}

	info := output.ServiceInfo

	for _, ignoreTimeKey := range []string{
		"startTime",
		"updateTime",
	} {
		delete(info, ignoreTimeKey)
	}

	return nil
}

func (c *Controller) SetUpstreamHandler(addFunc runtime.UpstreamHandlerAddFunc) error {
	return addFunc(KindName, c.updateFromEdge)
}
