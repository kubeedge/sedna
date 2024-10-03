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

package federatedlearning

import (
	"context"
	"encoding/json"
	"fmt"

	sednav1 "github.com/kubeedge/sedna/pkg/apis/sedna/v1alpha1"
	"github.com/kubeedge/sedna/pkg/globalmanager/runtime"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func (c *Controller) updateModelMetrics(jobName, namespace string, metrics []sednav1.Metric) error {
	var err error
	job, err := c.client.FederatedLearningJobs(namespace).Get(context.TODO(), jobName, metav1.GetOptions{})
	if err != nil {
		// federated crd not found
		return err
	}
	modelName := job.Spec.AggregationWorker.Model.Name
	client := c.client.Models(namespace)

	return runtime.RetryUpdateStatus(modelName, namespace, func() error {
		model, err := client.Get(context.TODO(), modelName, metav1.GetOptions{})
		if err != nil {
			return err
		}

		now := metav1.Now()
		model.Status.UpdateTime = &now
		model.Status.Metrics = metrics
		_, err = client.UpdateStatus(context.TODO(), model, metav1.UpdateOptions{})
		return err
	})
}

func (c *Controller) appendStatusCondition(name, namespace string, cond sednav1.FLJobCondition) error {
	client := c.client.FederatedLearningJobs(namespace)

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

// updateFromEdge updates the federated job's status
func (c *Controller) updateFromEdge(name, namespace, _ string, content []byte) (err error) {
	// JobInfo defines the job information
	type JobInfo struct {
		// Current training round
		CurrentRound int    `json:"currentRound"`
		UpdateTime   string `json:"updateTime"`
	}

	// Output defines job output information
	type Output struct {
		Models  []runtime.Model `json:"models"`
		JobInfo *JobInfo        `json:"ownerInfo"`
	}

	var status struct {
		Phase  string  `json:"phase"`
		Status string  `json:"status"`
		Output *Output `json:"output"`
	}

	err = json.Unmarshal(content, &status)
	if err != nil {
		return
	}

	output := status.Output

	if output != nil {
		// Update the model's metrics
		if len(output.Models) > 0 {
			// only one model
			model := output.Models[0]
			metrics := runtime.ConvertMapToMetrics(model.Metrics)
			if len(metrics) > 0 {
				c.updateModelMetrics(name, namespace, metrics)
			}
		}

		jobInfo := output.JobInfo
		// update job info if having any info
		if jobInfo != nil && jobInfo.CurrentRound > 0 {
			// Find a good place to save the progress info
			// TODO: more meaningful reason/message
			reason := "DoTraining"
			message := fmt.Sprintf("Round %v reaches at %s", jobInfo.CurrentRound, jobInfo.UpdateTime)
			cond := NewJobCondition(sednav1.FLJobCondTraining, reason, message)
			c.appendStatusCondition(name, namespace, cond)
		}
	}

	return nil
}

func (c *Controller) SetUpstreamHandler(addFunc runtime.UpstreamHandlerAddFunc) error {
	return addFunc(KindName, c.updateFromEdge)
}
