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
	"k8s.io/apimachinery/pkg/watch"

	sednav1 "github.com/kubeedge/sedna/pkg/apis/sedna/v1alpha1"
	"github.com/kubeedge/sedna/pkg/globalmanager/runtime"
)

func (c *Controller) syncToEdge(eventType watch.EventType, obj interface{}) error {
	job, ok := obj.(*sednav1.FederatedLearningJob)
	if !ok {
		return nil
	}

	// Since Kind may be empty,
	// we need to fix the kind here if missing.
	// more details at https://github.com/kubernetes/kubernetes/issues/3030
	job.Kind = KindName

	// broadcast to all nodes specified in spec
	nodeset := make(map[string]bool)
	for _, trainingWorker := range job.Spec.TrainingWorkers {
		// Here only propagate to the nodes with non empty name
		if len(trainingWorker.Template.Spec.NodeName) > 0 {
			nodeset[trainingWorker.Template.Spec.NodeName] = true
		}
	}

	for nodeName := range nodeset {
		c.sendToEdgeFunc(nodeName, eventType, job)
	}
	return nil
}

func (c *Controller) SetDownstreamSendFunc(f runtime.DownstreamSendFunc) error {
	c.sendToEdgeFunc = f

	return nil
}
