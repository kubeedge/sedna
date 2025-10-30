/*
Copyright 2025 The KubeEdge Authors.

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

package llmjointinference

import (
	appsv1 "k8s.io/api/apps/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"

	sednav1 "github.com/kubeedge/sedna/pkg/apis/sedna/v1alpha1"
)

// addDeployment enqueues the LLMJointInferenceService object when a deployment is created
func (c *Controller) addDeployment(obj interface{}) {
	d := obj.(*appsv1.Deployment)
	if d.DeletionTimestamp != nil {
		// On a restart of the controller manager, it's possible for an object to
		// show up in a state that is already pending deletion.
		c.deleteDeployment(d)
		return
	}
	if service := c.resolveControllerRef(d.Namespace, d.OwnerReferences); service != nil {
		c.enqueueController(service, false)
	}
}

// updateDeployment figures out what LLMJointInferenceService manages this Deployment and enqueues it
func (c *Controller) updateDeployment(old, cur interface{}) {
	oldD := old.(*appsv1.Deployment)
	curD := cur.(*appsv1.Deployment)
	if curD.ResourceVersion == oldD.ResourceVersion {
		// Periodic resync will send update events for all known Deployments.
		// Two different versions of the same Deployment will always have different RVs.
		return
	}

	if service := c.resolveControllerRef(curD.Namespace, curD.OwnerReferences); service != nil {
		c.enqueueController(service, false)
	}
}

// deleteDeployment enqueues the LLMJointInferenceService obj when a Deployment is deleted
func (c *Controller) deleteDeployment(obj interface{}) {
	d, ok := obj.(*appsv1.Deployment)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			klog.V(2).Infof("Couldn't get object from tombstone %#v", obj)
			return
		}
		d, ok = tombstone.Obj.(*appsv1.Deployment)
		if !ok {
			klog.V(2).Infof("Tombstone contained object that is not a Deployment %#v", obj)
			return
		}
	}

	if service := c.resolveControllerRef(d.Namespace, d.OwnerReferences); service != nil {
		c.enqueueController(service, false)
	}
}

// resolveControllerRef returns the controller referenced by a ControllerRef,
// or nil if the ControllerRef could not be resolved to a matching controller
// of the correct Kind.
func (c *Controller) resolveControllerRef(namespace string, controllerRef []metav1.OwnerReference) *sednav1.LLMJointInferenceService {
	// We can't look up by UID, so look up by Name and then verify UID.
	// Don't even try to look up by Name if it's the wrong Kind.
	if len(controllerRef) == 0 {
		return nil
	}
	ref := controllerRef[0]
	if ref.Kind != gvk.Kind {
		return nil
	}
	service, err := c.serviceLister.LLMJointInferenceServices(namespace).Get(ref.Name)
	if err != nil {
		return nil
	}
	if service.UID != ref.UID {
		// The controller we found with this Name is not the same one that the
		// ControllerRef points to.
		return nil
	}
	return service
}
