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

package lifelonglearning

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	sednav1 "github.com/kubeedge/sedna/pkg/apis/sedna/v1alpha1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"

	"github.com/kubeedge/sedna/pkg/globalmanager/runtime"
)

type Model = runtime.Model

// ConditionData the data of this condition including the input/output to do the next step
type ConditionData struct {
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

func (cd *ConditionData) joinModelURLs(model *Model, models []Model) []string {
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

func (cd *ConditionData) Unmarshal(data []byte) error {
	return json.Unmarshal(data, cd)
}

func (cd ConditionData) Marshal() ([]byte, error) {
	return json.Marshal(cd)
}

func (cd *ConditionData) GetInputModelURLs() []string {
	return cd.joinModelURLs(cd.Input.Model, cd.Input.Models)
}

func (cd *ConditionData) GetOutputModelURLs() []string {
	return cd.joinModelURLs(cd.Output.Model, cd.Output.Models)
}

func (c *Controller) appendStatusCondition(name, namespace string, cond sednav1.LLJobCondition) error {
	client := c.client.LifelongLearningJobs(namespace)
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

func (c *Controller) updateStatusKnowledgeBase(name, namespace string, cd ConditionData) error {
	client := c.client.LifelongLearningJobs(namespace)
	return runtime.RetryUpdateStatus(name, namespace, func() error {
		// check if models field exits
		if cd.Output == nil || cd.Output.Models == nil || len(cd.Output.Models) == 0 {
			klog.V(4).Infof("output models is nil, name is %s", name)
			return nil
		}

		numberOfSamples := 0
		aiModels := sednav1.AIModels{}
		aiModels.ListOfAIModels = make([]sednav1.AIModel, 0, len(cd.Output.Models))

		aiClasses := sednav1.AIClasses{}
		aiClasses.ListOfAIClasses = cd.Output.Models[0].Classes
		aiClasses.NumberOfAIClasses = len(aiClasses.ListOfAIClasses)

		for _, m := range cd.Output.Models {
			for modelName, metrics := range m.CurrentMetric {
				am := sednav1.AIModel{}
				am.ModelID = modelName
				// TODO: current only support showing sample number of one task instead of one model
				// consider to support showing sample number of one model based on the requirement.
				am.NumberOfTrainSamples = m.NumberOfLabeledUnseenSample

				for _, v := range metrics {
					// TODO: current only support one metric, consider to support multiple metrics.
					am.AveragePrecision = v
					break
				}
				aiModels.ListOfAIModels = append(aiModels.ListOfAIModels, am)
			}
			numberOfSamples += m.NumberOfLabeledUnseenSample
		}
		aiModels.NumberOfAIModels = len(aiModels.ListOfAIModels)

		samples := sednav1.Samples{
			NumberOfLabeledUnseenSample: numberOfSamples,
			NumberOfUnseenSample:        0,
		}

		kb := sednav1.KnowledgeBase{
			AIModels:  aiModels,
			AIClasses: aiClasses,
			Samples:   samples,
		}

		job, err := client.Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			klog.Errorf("client get name error, name is %s, error is %s", name, err)
			return err
		}

		job.Status.KnowledgeBase = kb
		_, err = client.UpdateStatus(context.TODO(), job, metav1.UpdateOptions{})

		if err != nil {
			klog.Errorf("update status error, err msg is %v", err)
			return err
		}

		return nil
	})
}

// updateFromEdge syncs the edge updates to k8s
func (c *Controller) updateFromEdge(name, namespace, _ string, content []byte) error {
	var jobStatus struct {
		Phase  string `json:"phase"`
		Status string `json:"status"`
	}

	err := json.Unmarshal(content, &jobStatus)
	if err != nil {
		klog.Errorf("json unmarshal error, content is %s, err is %v", content, err)
		return err
	}

	// Get the condition data.
	// Here unmarshal and marshal immediately to skip the unnecessary fields
	var condData ConditionData
	err = json.Unmarshal(content, &condData)
	if err != nil {
		klog.Errorf("json unmarshal error, content is %s, err is %v", content, err)
		return err
	}

	condDataBytes, _ := json.Marshal(&condData)

	cond := sednav1.LLJobCondition{
		Status:             v1.ConditionTrue,
		LastHeartbeatTime:  metav1.Now(),
		LastTransitionTime: metav1.Now(),
		Data:               string(condDataBytes),
		Message:            "reported by lc",
	}

	switch strings.ToLower(jobStatus.Phase) {
	case "train":
		cond.Stage = sednav1.LLJobTrain
	case "eval":
		cond.Stage = sednav1.LLJobEval
	case "deploy":
		cond.Stage = sednav1.LLJobDeploy
	default:
		return fmt.Errorf("invalid condition stage: %v", jobStatus.Phase)
	}

	switch strings.ToLower(jobStatus.Status) {
	case "ready":
		cond.Type = sednav1.LLJobStageCondReady
	case "completed":
		cond.Type = sednav1.LLJobStageCondCompleted
	case "failed":
		cond.Type = sednav1.LLJobStageCondFailed
	case "waiting":
		cond.Type = sednav1.LLJobStageCondWaiting
	default:
		return fmt.Errorf("invalid condition type: %v", jobStatus.Status)
	}

	err = c.updateStatusKnowledgeBase(name, namespace, condData)
	if err != nil {
		klog.Errorf("failed to update KnowledgeBase, err:%w", err)
		return err
	}

	err = c.appendStatusCondition(name, namespace, cond)
	if err != nil {
		klog.Errorf("failed to append condition, err:%w", err)
		return err
	}
	return nil
}

func (c *Controller) SetUpstreamHandler(addFunc runtime.UpstreamHandlerAddFunc) error {
	return addFunc(KindName, c.updateFromEdge)
}
