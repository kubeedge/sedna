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

	ann := job.GetAnnotations()
	if ann != nil {
		trainNodeName = ann[runtime.AnnotationsKeyPrefix+string(sednav1.ILJobTrain)]
		evalNodeName = ann[runtime.AnnotationsKeyPrefix+string(sednav1.ILJobEval)]
	}

	if eventType == watch.Deleted {
		// delete jobs from all LCs
		for _, v := range []string{dsNodeName, trainNodeName, evalNodeName} {
			if v != "" {
				c.sendToEdgeFunc(v, eventType, job)
			}
		}
		return nil
	}

	latestCondition := jobConditions[len(jobConditions)-1]
	currentType := latestCondition.Type
	jobStage := latestCondition.Stage

	syncModelWithName := func(modelName string) {
		if err := c.syncModelWithName(dsNodeName, modelName, job.Namespace); err != nil {
			klog.Warningf("Error to sync model %s when sync incremental learning job %s to node %s: %v",
				modelName, job.Name, dsNodeName, err)
		}
	}

	syncJobWithNodeName := func(nodeName string) {
		if err := c.sendToEdgeFunc(nodeName, eventType, job); err != nil {
			klog.Warningf("Error to sync incremental learning job %s to node %s in stage %s: %v",
				job.Name, nodeName, jobStage, err)
		}
	}

	runtime.InjectSecretAnnotations(c.kubeClient, job, job.Spec.CredentialName)

	doJobStageEvent := func(modelName string, nodeName string) {
		if currentType == sednav1.ILJobStageCondWaiting {
			syncJobWithNodeName(dsNodeName)
			syncModelWithName(modelName)
		} else if currentType == sednav1.ILJobStageCondRunning {
			if nodeName != "" {
				syncJobWithNodeName(nodeName)
			}
		} else if currentType == sednav1.ILJobStageCondCompleted || currentType == sednav1.ILJobStageCondFailed {
			if nodeName != dsNodeName {
				// delete LC's job from nodeName that's different from dataset node when worker's status is completed or failed.
				c.sendToEdgeFunc(nodeName, watch.Deleted, job)
			}
		}
	}

	switch jobStage {
	case sednav1.ILJobTrain:
		doJobStageEvent(job.Spec.InitialModel.Name, trainNodeName)
	case sednav1.ILJobEval:
		doJobStageEvent(job.Spec.DeploySpec.Model.Name, evalNodeName)
	}

	return nil
}

func (c *Controller) SetDownstreamSendFunc(f runtime.DownstreamSendFunc) error {
	c.sendToEdgeFunc = f
	return nil
}
