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

package incrementallearning

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	sednav1 "github.com/kubeedge/sedna/pkg/apis/sedna/v1alpha1"
	"github.com/kubeedge/sedna/pkg/globalmanager/runtime"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type Model = runtime.Model

// IncrementalCondData the data of this condition including the input/output to do the next step
type IncrementalCondData struct {
	Input *struct {
		// Only one model cases
		Model  *Model  `json:"model,omitempty"`
		Models []Model `json:"models,omitempty"`

		DataURL string `json:"dataURL,omitempty"`

		// the data samples reference will be stored into this URL.
		// The content of this url would be:
		// # the first uncomment line means the directory
		// s3://dataset/
		// mnist/0.jpg
		// mnist/1.jpg
		DataIndexURL string `json:"dataIndexURL,omitempty"`

		OutputDir string `json:"outputDir,omitempty"`
	} `json:"input,omitempty"`

	Output *struct {
		Model  *Model  `json:"model,omitempty"`
		Models []Model `json:"models,omitempty"`
	} `json:"output,omitempty"`
}

func (cd *IncrementalCondData) joinModelURLs(model *Model, models []Model) []string {
	var modelURLs []string
	if model != nil {
		modelURLs = append(modelURLs, model.GetURL())
	} else {
		for _, m := range models {
			modelURLs = append(modelURLs, m.GetURL())
		}
	}
	return modelURLs
}

func (cd *IncrementalCondData) GetInputModelURLs() []string {
	return cd.joinModelURLs(cd.Input.Model, cd.Input.Models)
}

func (cd *IncrementalCondData) GetOutputModelURLs() []string {
	return cd.joinModelURLs(cd.Output.Model, cd.Output.Models)
}

func (cd *IncrementalCondData) Unmarshal(data []byte) error {
	return json.Unmarshal(data, cd)
}

func (cd IncrementalCondData) Marshal() ([]byte, error) {
	return json.Marshal(cd)
}

func (c *Controller) appendStatusCondition(name, namespace string, cond sednav1.ILJobCondition) error {
	client := c.client.IncrementalLearningJobs(namespace)
	return runtime.RetryUpdateStatus(name, namespace, func() error {
		job, err := client.Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		job.Status.Conditions = append(job.Status.Conditions, cond)
		_, err = client.UpdateStatus(context.TODO(), job, metav1.UpdateOptions{})
		return err
	})
}

// updateFromEdge syncs the edge updates to k8s
func (c *Controller) updateFromEdge(name, namespace, operation string, content []byte) error {
	var jobStatus struct {
		Phase  string `json:"phase"`
		Status string `json:"status"`
	}

	err := json.Unmarshal(content, &jobStatus)
	if err != nil {
		return err
	}

	// Get the condition data.
	// Here unmarshal and marshal immediately to skip the unnecessary fields
	var condData IncrementalCondData
	err = json.Unmarshal(content, &condData)
	if err != nil {
		return err
	}
	condDataBytes, _ := json.Marshal(&condData)

	cond := sednav1.ILJobCondition{
		Status:             v1.ConditionTrue,
		LastHeartbeatTime:  metav1.Now(),
		LastTransitionTime: metav1.Now(),
		Data:               string(condDataBytes),
		Message:            "reported by lc",
	}

	switch strings.ToLower(jobStatus.Phase) {
	case "train":
		cond.Stage = sednav1.ILJobTrain
	case "eval":
		cond.Stage = sednav1.ILJobEval
	case "deploy":
		cond.Stage = sednav1.ILJobDeploy
	default:
		return fmt.Errorf("invalid condition stage: %v", jobStatus.Phase)
	}

	switch strings.ToLower(jobStatus.Status) {
	case "ready":
		cond.Type = sednav1.ILJobStageCondReady
	case "completed":
		cond.Type = sednav1.ILJobStageCondCompleted
	case "failed":
		cond.Type = sednav1.ILJobStageCondFailed
	case "waiting":
		cond.Type = sednav1.ILJobStageCondWaiting
	default:
		return fmt.Errorf("invalid condition type: %v", jobStatus.Status)
	}

	err = c.appendStatusCondition(name, namespace, cond)
	if err != nil {
		return fmt.Errorf("failed to append condition, err:%w", err)
	}
	return nil
}

func (c *Controller) SetUpstreamHandler(addFunc runtime.UpstreamHandlerAddFunc) error {
	return addFunc(KindName, c.updateFromEdge)
}
