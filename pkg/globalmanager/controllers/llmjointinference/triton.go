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
	"context"
	"fmt"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/klog/v2"

	sednav1 "github.com/kubeedge/sedna/pkg/apis/sedna/v1alpha1"
)

const (
	// ConfigMapName is the name of ConfigMap containing inference engine images
	ConfigMapName = "inference-engine-config"
)

// getTritonImage gets Triton image from inference-engine-config ConfigMap
func (c *Controller) getTritonImage(namespace string) (string, error) {
	cm, err := c.kubeClient.CoreV1().ConfigMaps(namespace).Get(context.TODO(), ConfigMapName, metav1.GetOptions{})
	if err != nil {
		return "", fmt.Errorf("failed to get ConfigMap %s: %v", ConfigMapName, err)
	}

	tritonImage, ok := cm.Data["triton"]
	if !ok {
		return "", fmt.Errorf("triton image not found in ConfigMap %s", ConfigMapName)
	}

	return tritonImage, nil
}

func (c *Controller) syncTritonDeployment(service *sednav1.LLMJointInferenceService) error {
	tritonImage, err := c.getTritonImage(service.Namespace)
	if err != nil {
		return fmt.Errorf("failed to get Triton image: %v", err)
	}

	name := fmt.Sprintf("%s-triton", service.Name)

	deployment := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: service.Namespace,
			Labels: map[string]string{
				"app":               name,
				"llmjointinference": service.Name,
			},
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(service, gvk),
			},
		},
		Spec: appsv1.DeploymentSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					"app": name,
				},
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"app": name,
					},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "triton",
							Image: tritonImage, // Will be replaced with ConfigMap value
							Args: []string{
								"tritonserver",
								"--model-repository=/models",
								"--allow-gpu-metrics=true",
								"--allow-metrics=true",
							},
							Ports: []v1.ContainerPort{
								{Name: "http", ContainerPort: TritonHTTPPort, Protocol: v1.ProtocolTCP},
								{Name: "grpc", ContainerPort: TritonGRPCPort, Protocol: v1.ProtocolTCP},
								{Name: "metrics", ContainerPort: TritonMetricsPort, Protocol: v1.ProtocolTCP},
							},
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("4"),
									v1.ResourceMemory: resource.MustParse("8Gi"),
									"nvidia.com/gpu":  resource.MustParse("1"),
								},
								Limits: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("8"),
									v1.ResourceMemory: resource.MustParse("16Gi"),
									"nvidia.com/gpu":  resource.MustParse("1"),
								},
							},
							VolumeMounts: []v1.VolumeMount{
								{Name: "model-repository", MountPath: "/models"},
							},
							LivenessProbe: &v1.Probe{
								ProbeHandler: v1.ProbeHandler{
									HTTPGet: &v1.HTTPGetAction{
										Path: "/v2/health/live",
										Port: intstr.FromInt32(TritonHTTPPort),
									},
								},
								InitialDelaySeconds: 30,
								PeriodSeconds:       5,
							},
							ReadinessProbe: &v1.Probe{
								ProbeHandler: v1.ProbeHandler{
									HTTPGet: &v1.HTTPGetAction{
										Path: "/v2/health/ready",
										Port: intstr.FromInt32(TritonHTTPPort),
									},
								},
								InitialDelaySeconds: 30,
								PeriodSeconds:       5,
							},
						},
					},
					Volumes: []v1.Volume{
						{
							Name: "model-repository",
							VolumeSource: v1.VolumeSource{
								PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
									ClaimName: fmt.Sprintf("%s-models", name),
								},
							},
						},
					},
				},
			},
		},
	}

	if service.Spec.CloudWorker.Template.Spec.Containers != nil {
		deployment.Spec.Template.Spec = service.Spec.CloudWorker.Template.Spec
	}

	existing, err := c.deploymentsLister.Deployments(service.Namespace).Get(name)
	if err != nil {
		if errors.IsNotFound(err) {
			_, err = c.kubeClient.AppsV1().Deployments(service.Namespace).Create(context.TODO(), deployment, metav1.CreateOptions{})
			if err != nil {
				return fmt.Errorf("failed to create Deployment: %v", err)
			}
			klog.V(2).Infof("Created Deployment %s/%s", service.Namespace, name)
		}
		return err
	}

	existing.Spec = deployment.Spec
	_, err = c.kubeClient.AppsV1().Deployments(service.Namespace).Update(context.TODO(), existing, metav1.UpdateOptions{})
	if err != nil {
		return fmt.Errorf("failed to update Deployment: %v", err)
	}
	klog.V(2).Infof("Updated Deployment %s/%s", service.Namespace, name)

	return nil
}

func (c *Controller) syncTritonService(service *sednav1.LLMJointInferenceService) error {
	name := fmt.Sprintf("%s-triton", service.Name)

	svc := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: service.Namespace,
			Labels: map[string]string{
				"app":               name,
				"llmjointinference": service.Name,
			},
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(service, gvk),
			},
		},
		Spec: v1.ServiceSpec{
			Type: v1.ServiceTypeClusterIP,
			Ports: []v1.ServicePort{
				{
					Name:       "http",
					Port:       TritonHTTPPort,
					Protocol:   v1.ProtocolTCP,
					TargetPort: intstr.FromInt32(TritonHTTPPort),
				},
				{
					Name:       "grpc",
					Port:       TritonGRPCPort,
					Protocol:   v1.ProtocolTCP,
					TargetPort: intstr.FromInt32(TritonGRPCPort),
				},
				{
					Name:       "metrics",
					Port:       TritonMetricsPort,
					Protocol:   v1.ProtocolTCP,
					TargetPort: intstr.FromInt32(TritonMetricsPort),
				},
			},
			Selector: map[string]string{
				"app": name,
			},
		},
	}

	// Create or update Service
	existing, err := c.kubeClient.CoreV1().Services(service.Namespace).Get(context.TODO(), name, metav1.GetOptions{})
	if err != nil {
		if errors.IsNotFound(err) {
			_, err = c.kubeClient.CoreV1().Services(service.Namespace).Create(context.TODO(), svc, metav1.CreateOptions{})
			if err != nil {
				return fmt.Errorf("failed to create Service: %v", err)
			}
			klog.V(2).Infof("Created Service %s/%s", service.Namespace, name)
		}
		return err
	}

	// Update existing Service
	existing.Spec = svc.Spec
	_, err = c.kubeClient.CoreV1().Services(service.Namespace).Update(context.TODO(), existing, metav1.UpdateOptions{})
	if err != nil {
		return fmt.Errorf("failed to update Service: %v", err)
	}
	klog.V(2).Infof("Updated Service %s/%s", service.Namespace, name)

	return nil
}

func (c *Controller) updateStatus(service *sednav1.LLMJointInferenceService) error {
	// Get latest service state
	latest, err := c.serviceLister.LLMJointInferenceServices(service.Namespace).Get(service.Name)
	if err != nil {
		return err
	}

	// Check Deployment status
	name := fmt.Sprintf("%s-triton", service.Name)
	deployment, err := c.deploymentsLister.Deployments(service.Namespace).Get(name)
	if err != nil {
		if !errors.IsNotFound(err) {
			return err
		}
		// Deployment not found, service is pending
		c.recorder.Event(service, v1.EventTypeNormal, "Pending", "Waiting for Deployment to be created")
		latest.Status.Conditions = []sednav1.LLMJointInferenceServiceCondition{
			{
				Type:    sednav1.LLMJointInferenceServiceCondPending,
				Status:  v1.ConditionTrue,
				Reason:  "DeploymentNotFound",
				Message: "Waiting for Deployment to be created",
			},
		}
	} else {
		// Update status based on Deployment state
		if deployment.Status.ReadyReplicas == *deployment.Spec.Replicas {
			latest.Status.Conditions = []sednav1.LLMJointInferenceServiceCondition{
				{
					Type:    sednav1.LLMJointInferenceServiceCondRunning,
					Status:  v1.ConditionTrue,
					Reason:  "DeploymentReady",
					Message: "All replicas are ready",
				},
			}
			latest.Status.Active = deployment.Status.ReadyReplicas
		} else {
			latest.Status.Conditions = []sednav1.LLMJointInferenceServiceCondition{
				{
					Type:    sednav1.LLMJointInferenceServiceCondPending,
					Status:  v1.ConditionTrue,
					Reason:  "DeploymentNotReady",
					Message: fmt.Sprintf("Waiting for all replicas to be ready (%d/%d)", deployment.Status.ReadyReplicas, *deployment.Spec.Replicas),
				},
			}
			latest.Status.Active = deployment.Status.ReadyReplicas
		}
	}

	// Update status
	_, err = c.client.LLMJointInferenceServices(service.Namespace).UpdateStatus(context.TODO(), latest, metav1.UpdateOptions{})
	if err != nil {
		return fmt.Errorf("failed to update status: %v", err)
	}

	return nil
}
