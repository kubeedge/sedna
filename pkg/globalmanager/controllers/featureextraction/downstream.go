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
	"fmt"

	"k8s.io/apimachinery/pkg/watch"

	sednav1 "github.com/kubeedge/sedna/pkg/apis/sedna/v1alpha1"
	"github.com/kubeedge/sedna/pkg/globalmanager/runtime"
)

func (c *Controller) syncToEdge(eventType watch.EventType, obj interface{}) error {
	service, ok := obj.(*sednav1.FeatureExtractionService)
	if !ok {
		return nil
	}

	// Since Kind may be empty,
	// we need to fix the kind here if missing.
	// more details at https://github.com/kubernetes/kubernetes/issues/3030
	service.Kind = KindName

	// Here only propagate to the nodes with non empty name
	nodeName := service.Spec.Template.Spec.NodeName
	if len(nodeName) == 0 {
		return fmt.Errorf("empty node name")
	}

	return c.sendToEdgeFunc(nodeName, eventType, service)
}

func (c *Controller) SetDownstreamSendFunc(f runtime.DownstreamSendFunc) error {
	c.sendToEdgeFunc = f
	return nil
}
