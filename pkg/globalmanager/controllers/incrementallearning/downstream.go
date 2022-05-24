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
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/klog/v2"

	sednav1 "github.com/kubeedge/sedna/pkg/apis/sedna/v1alpha1"
	"github.com/kubeedge/sedna/pkg/globalmanager/runtime"
)

// syncModelWithName will sync the model to the specified node.
// Now called when creating the incrementaljob.
func (c *Controller) syncModelWithName(nodeName, modelName, namespace string) error {
	model, err := c.client.Models(namespace).Get(context.TODO(), modelName, metav1.GetOptions{})
	if err != nil {
		// TODO: maybe use err.ErrStatus.Code == 404
		return fmt.Errorf("model(%s/%s) not found", namespace, modelName)
	}

	// Since model.Kind may be empty,
	// we need to fix the kind here if missing.
	// more details at https://github.com/kubernetes/kubernetes/issues/3030
	if len(model.Kind) == 0 {
		model.Kind = "Model"
	}

	runtime.InjectSecretAnnotations(c.kubeClient, model, model.Spec.CredentialName)

	c.sendToEdgeFunc(nodeName, watch.Added, model)
	return nil
}

func (c *Controller) syncToEdge(eventType watch.EventType, obj interface{}) error {
	job, ok := obj.(*sednav1.IncrementalLearningJob)
	if !ok {
		return nil
	}

	// Since Kind may be empty,
	// we need to fix the kind here if missing.
	// more details at https://github.com/kubernetes/kubernetes/issues/3030
	job.Kind = KindName

	jobConditions := job.Status.Conditions
	if len(jobConditions) == 0 {
		return nil
	}

	dataName := job.Spec.Dataset.Name
	ds, err := c.client.Datasets(job.Namespace).Get(context.TODO(), dataName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("dataset(%s/%s) not found", job.Namespace, dataName)
	}
	// LC has dataset object on this node that may call dataset node
	dsNodeName := ds.Spec.NodeName

	var trainNodeName string
	var evalNodeName string
	var deployNodeName string

	getAnnotationsNodeName := func(nodeName sednav1.ILJobStage) string {
		return runtime.AnnotationsKeyPrefix + string(nodeName)
	}
	ann := job.GetAnnotations()
	if ann != nil {
		trainNodeName = ann[getAnnotationsNodeName(sednav1.ILJobTrain)]
		evalNodeName = ann[getAnnotationsNodeName(sednav1.ILJobEval)]
		if _, ok := ann[runtime.ModelHotUpdateAnnotationsKey]; ok {
			deployNodeName = ann[getAnnotationsNodeName(sednav1.ILJobDeploy)]
		}
	}

	if eventType == watch.Deleted {
		// delete jobs from all LCs
		nodes := sets.NewString(dsNodeName, trainNodeName, evalNodeName, deployNodeName)

		for node := range nodes {
			c.sendToEdgeFunc(node, eventType, job)
		}

		return nil
	}

	latestCondition := jobConditions[len(jobConditions)-1]
	currentType := latestCondition.Type
	jobStage := latestCondition.Stage

	syncModelWithName := func(modelName string, nodeName string) {
		if err := c.syncModelWithName(nodeName, modelName, job.Namespace); err != nil {
			klog.Warningf("Error to sync model %s when sync incremental learning job %s to node %s: %v",
				modelName, job.Name, nodeName, err)
		}
	}

	syncJobWithNodeName := func(nodeName string) {
		if err := c.sendToEdgeFunc(nodeName, eventType, job); err != nil {
			klog.Warningf("Error to sync incremental learning job %s to node %s in stage %s: %v",
				job.Name, nodeName, jobStage, err)
		}
	}

	runtime.InjectSecretAnnotations(c.kubeClient, job, job.Spec.CredentialName)

	// isJobResidentNode checks whether nodeName is a job resident node
	isJobResidentNode := func(nodeName string) bool {
		// the node where LC monitors dataset and the node where inference worker is running are job resident node
		if nodeName == dsNodeName || nodeName == deployNodeName {
			return true
		}
		return false
	}

	// delete job
	deleteJob := func(nodeName string) {
		if !isJobResidentNode(nodeName) {
			// delete LC's job from nodeName that's different from dataset node when worker's status
			// is completed or failed.
			c.sendToEdgeFunc(nodeName, watch.Deleted, job)
		}
	}

	switch currentType {
	case sednav1.ILJobStageCondWaiting:
		switch jobStage {
		case sednav1.ILJobTrain:
			syncModelWithName(job.Spec.InitialModel.Name, dsNodeName)
			syncJobWithNodeName(dsNodeName)
		case sednav1.ILJobEval:
			syncModelWithName(job.Spec.DeploySpec.Model.Name, dsNodeName)
			if job.Spec.EvalSpec.InitialModel != nil {
				syncModelWithName(job.Spec.EvalSpec.InitialModel.Name, dsNodeName)
			}
			syncJobWithNodeName(dsNodeName)
		case sednav1.ILJobDeploy:
			deployNodeName = evalNodeName

			syncModelWithName(job.Spec.DeploySpec.Model.Name, evalNodeName)
			if job.Spec.EvalSpec.InitialModel != nil && !job.Spec.DeploySpec.Model.HotUpdateEnabled {
				syncModelWithName(job.Spec.EvalSpec.InitialModel.Name, deployNodeName)
			}
			syncJobWithNodeName(deployNodeName)
		}
	case sednav1.ILJobStageCondRunning:
		switch jobStage {
		case sednav1.ILJobTrain:
			syncJobWithNodeName(trainNodeName)
		case sednav1.ILJobEval:
			if trainNodeName != evalNodeName && trainNodeName != dsNodeName {
				c.sendToEdgeFunc(trainNodeName, watch.Deleted, job)
			}
			syncJobWithNodeName(evalNodeName)
		case sednav1.ILJobDeploy:
			if evalNodeName != deployNodeName && evalNodeName != dsNodeName {
				c.sendToEdgeFunc(evalNodeName, watch.Deleted, job)
			}

			if job.Spec.EvalSpec.InitialModel != nil {
				syncModelWithName(job.Spec.EvalSpec.InitialModel.Name, deployNodeName)
			}
			syncModelWithName(job.Spec.DeploySpec.Model.Name, deployNodeName)
			syncJobWithNodeName(deployNodeName)
		}
	case sednav1.ILJobStageCondCompleted, sednav1.ILJobStageCondFailed:
		if !job.Spec.DeploySpec.Model.HotUpdateEnabled {
			deployNodeName = evalNodeName
		}
		switch jobStage {
		case sednav1.ILJobTrain:
			deleteJob(trainNodeName)
		case sednav1.ILJobEval:
			deleteJob(evalNodeName)
		case sednav1.ILJobDeploy:
			deleteJob(deployNodeName)
		}
	}

	return nil
}

func (c *Controller) SetDownstreamSendFunc(f runtime.DownstreamSendFunc) error {
	c.sendToEdgeFunc = f
	return nil
}
