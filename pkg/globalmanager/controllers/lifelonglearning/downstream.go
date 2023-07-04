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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/klog/v2"

	sednav1 "github.com/kubeedge/sedna/pkg/apis/sedna/v1alpha1"
	"github.com/kubeedge/sedna/pkg/globalmanager/runtime"
)

func (c *Controller) syncToEdge(eventType watch.EventType, obj interface{}) error {
	job, ok := obj.(*sednav1.LifelongLearningJob)
	if !ok {
		klog.V(4).Infof("get job %s failed, stop to sync to edge", job.Name)
		return nil
	}

	// Since Kind may be empty,
	// we need to fix the kind here if missing.
	// more details at https://github.com/kubernetes/kubernetes/issues/3030
	job.Kind = KindName

	dataName := job.Spec.Dataset.Name
	// LC has dataset object on this node that may call dataset node
	var dsNodeName string
	ds, err := c.client.Datasets(job.Namespace).Get(context.TODO(), dataName, metav1.GetOptions{})
	if err != nil {
		klog.Errorf("not found job(name=%s/%s)'s dataset, error: %v", job.Kind, job.Name, err)
	} else {
		dsNodeName = ds.Spec.NodeName
	}

	var trainNodeName string
	var evalNodeName string
	var deployNodeName string

	getAnnotationsNodeName := func(nodeName sednav1.LLJobStage) string {
		klog.V(4).Infof("getAnnotationsNodeName return %s", job.Name)
		return runtime.AnnotationsKeyPrefix + string(nodeName)
	}
	ann := job.GetAnnotations()
	if ann != nil {
		trainNodeName = ann[getAnnotationsNodeName(sednav1.LLJobTrain)]
		evalNodeName = ann[getAnnotationsNodeName(sednav1.LLJobEval)]
		deployNodeName = ann[getAnnotationsNodeName(sednav1.LLJobDeploy)]
	}

	if eventType == watch.Deleted {
		// delete jobs from all LCs
		nodes := sets.NewString(dsNodeName, trainNodeName, evalNodeName, deployNodeName)

		for node := range nodes {
			c.sendToEdgeFunc(node, eventType, job)
		}

		return nil
	}

	if dsNodeName == "" {
		return nil
	}

	jobConditions := job.Status.Conditions
	if len(jobConditions) == 0 {
		return nil
	}

	latestCondition := jobConditions[len(jobConditions)-1]
	currentType := latestCondition.Type
	jobStage := latestCondition.Stage

	syncJobWithNodeName := func(nodeName string) {
		klog.V(4).Infof("syncJobWithNodeName nodeName is %s, job name is %s ", nodeName, job.Name)
		if err := c.sendToEdgeFunc(nodeName, eventType, job); err != nil {
			klog.Warningf("Error to sync lifelong learning job %s to node %s in stage %s: %v",
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

	doJobStageEvent := func(nodeName string) {
		switch currentType {
		case sednav1.LLJobStageCondWaiting:
			klog.V(4).Infof("LLJobStageCondWaiting, dataset nodeName is %s", nodeName)
			syncJobWithNodeName(dsNodeName)
		case sednav1.LLJobStageCondRunning:
			klog.V(4).Infof("LLJobStageCondRunning, nodeName is %s", nodeName)
			syncJobWithNodeName(nodeName)
		case sednav1.LLJobStageCondCompleted, sednav1.LLJobStageCondFailed:
			klog.V(4).Infof("LLJobStageCondCompleted, nodeName is %s", nodeName)
			if !isJobResidentNode(nodeName) {
				// delete LC's job from nodeName that's different from dataset node when worker's status
				// is completed or failed.
				c.sendToEdgeFunc(nodeName, watch.Deleted, job)
			}
		}
	}

	switch jobStage {
	case sednav1.LLJobTrain:
		doJobStageEvent(trainNodeName)
	case sednav1.LLJobEval:
		doJobStageEvent(evalNodeName)
	case sednav1.LLJobDeploy:
		doJobStageEvent(deployNodeName)
	}

	return nil
}

func (c *Controller) SetDownstreamSendFunc(f runtime.DownstreamSendFunc) error {
	c.sendToEdgeFunc = f
	return nil
}
