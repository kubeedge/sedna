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

package globalmanager

import (
	"context"
	"fmt"
	"math"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	k8scontroller "k8s.io/kubernetes/pkg/controller"
)

const (
	// DefaultBackOff is the default backoff period
	DefaultBackOff = 10 * time.Second
	// MaxBackOff is the max backoff period
	MaxBackOff          = 360 * time.Second
	statusUpdateRetries = 3
	// setting some prefix for container path, include data and code prefix
	codePrefix         = "/home/work"
	dataPrefix         = "/home/data"
	bigModelPort int32 = 5000
)

// CreateVolumeMap creates volumeMap for container and
// returns volumeMounts and volumes for stage of creating pod
func CreateVolumeMap(containerPara *ContainerPara) ([]v1.VolumeMount, []v1.Volume) {
	var volumeMounts []v1.VolumeMount
	var volumes []v1.Volume
	volumetype := v1.HostPathDirectory
	mountPathMap := make(map[string]bool)
	duplicateIdx := make(map[int]bool)
	for i, v := range containerPara.volumeMountList {
		if mountPathMap[v] {
			duplicateIdx[i] = true
			continue
		}
		mountPathMap[v] = true
		tempVolumeMount := v1.VolumeMount{
			MountPath: v,
			Name:      containerPara.volumeMapName[i],
		}
		volumeMounts = append(volumeMounts, tempVolumeMount)
	}
	for i, v := range containerPara.volumeList {
		if duplicateIdx[i] {
			continue
		}
		tempVolume := v1.Volume{
			Name: containerPara.volumeMapName[i],
			VolumeSource: v1.VolumeSource{
				HostPath: &v1.HostPathVolumeSource{
					Path: v,
					Type: &volumetype,
				},
			},
		}
		volumes = append(volumes, tempVolume)
	}
	return volumeMounts, volumes
}

// CreateEnvVars creates EnvMap for container
// include EnvName and EnvValue map for stage of creating a pod
func CreateEnvVars(envMap map[string]string) []v1.EnvVar {
	var envVars []v1.EnvVar
	for envName, envValue := range envMap {
		Env := v1.EnvVar{
			Name:  envName,
			Value: envValue,
		}
		envVars = append(envVars, Env)
	}
	return envVars
}

// MatchContainerBaseImage searches the base image
func MatchContainerBaseImage(imageHub map[string]string, frameName string, frameVersion string) (string, error) {
	inputImageName := frameName + ":" + frameVersion
	for imageName, imageURL := range imageHub {
		if inputImageName == imageName {
			return imageURL, nil
		}
	}
	return "", fmt.Errorf("image %v not exists in imagehub", inputImageName)
}

// GetNodeIPByName get node ip by node name
func GetNodeIPByName(kubeClient kubernetes.Interface, name string) (string, error) {
	n, err := kubeClient.CoreV1().Nodes().Get(context.Background(), name, metav1.GetOptions{})
	if err != nil {
		return "", err
	}

	typeToAddress := make(map[v1.NodeAddressType]string)
	for _, addr := range n.Status.Addresses {
		typeToAddress[addr.Type] = addr.Address
	}
	address, found := typeToAddress[v1.NodeExternalIP]
	if found {
		return address, nil
	}

	address, found = typeToAddress[v1.NodeInternalIP]
	if found {
		return address, nil
	}
	return "", fmt.Errorf("can't found node ip for node %s", name)
}

// GenerateLabels generates labels for an object
func GenerateLabels(object CommonInterface) map[string]string {
	kind := object.GroupVersionKind().Kind
	group := object.GroupVersionKind().Group
	name := object.GetName()
	key := strings.ToLower(kind) + "." + group + "/name"
	labels := make(map[string]string)
	labels[key] = name
	return labels
}

// GenerateSelector generates selector for an object
func GenerateSelector(object CommonInterface) (labels.Selector, error) {
	ls := &metav1.LabelSelector{
		MatchLabels: GenerateLabels(object),
	}
	return metav1.LabelSelectorAsSelector(ls)
}

// CreateKubernetesService creates a k8s service for an object given ip and port
func CreateKubernetesService(kubeClient kubernetes.Interface, object CommonInterface, inputPort int32, inputIP string) (int32, error) {
	ctx := context.Background()
	name := object.GetName()
	namespace := object.GetNamespace()
	kind := object.GroupVersionKind().Kind
	targePort := intstr.IntOrString{
		IntVal: inputPort,
	}
	serviceSpec := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Namespace:    object.GetNamespace(),
			GenerateName: name + "-" + "service" + "-",
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(object, object.GroupVersionKind()),
			},
			Labels: GenerateLabels(object),
		},
		Spec: v1.ServiceSpec{
			Selector: GenerateLabels(object),
			ExternalIPs: []string{
				inputIP,
			},
			Type: v1.ServiceTypeNodePort,
			Ports: []v1.ServicePort{
				{
					Port:       inputPort,
					TargetPort: targePort,
				},
			},
		},
	}
	service, err := kubeClient.CoreV1().Services(namespace).Create(ctx, serviceSpec, metav1.CreateOptions{})
	if err != nil {
		klog.Warningf("failed to create service for %v %v/%v, err:%s", kind, namespace, name, err)
		return 0, err
	}

	klog.V(2).Infof("Service %s is created successfully for %v %v/%v", service.Name, kind, namespace, name)
	return service.Spec.Ports[0].NodePort, nil
}

// getBackoff calc the next wait time for the key
func getBackoff(queue workqueue.RateLimitingInterface, key interface{}) time.Duration {
	exp := queue.NumRequeues(key)

	if exp <= 0 {
		return time.Duration(0)
	}

	// The backoff is capped such that 'calculated' value never overflows.
	backoff := float64(DefaultBackOff.Nanoseconds()) * math.Pow(2, float64(exp-1))
	if backoff > math.MaxInt64 {
		return MaxBackOff
	}

	calculated := time.Duration(backoff)
	if calculated > MaxBackOff {
		return MaxBackOff
	}
	return calculated
}

func calcActivePodCount(pods []*v1.Pod) int32 {
	var result int32 = 0
	for _, p := range pods {
		if v1.PodSucceeded != p.Status.Phase &&
			v1.PodFailed != p.Status.Phase &&
			p.DeletionTimestamp == nil {
			result++
		}
	}
	return result
}

func injectContainerPara(pod *v1.Pod, containerPara *ContainerPara, object CommonInterface) {

	// inject our predefined volumes/envs
	volumeMounts, volumes := CreateVolumeMap(containerPara)
	envs := CreateEnvVars(containerPara.env)
	pod.Spec.Volumes = append(pod.Spec.Volumes, volumes...)
	for idx := range pod.Spec.Containers {
		pod.Spec.Containers[idx].Env = append(
			pod.Spec.Containers[idx].Env, envs...,
		)
		pod.Spec.Containers[idx].VolumeMounts = append(
			pod.Spec.Containers[idx].VolumeMounts, volumeMounts...,
		)
	}

	if pod.Labels == nil {
		pod.Labels = make(map[string]string)
	}
	for k, v := range GenerateLabels(object) {
		pod.Labels[k] = v
	}

	pod.GenerateName = object.GetName() + "-" + strings.ToLower(containerPara.workerType) + "-"

	pod.Namespace = object.GetNamespace()

	if containerPara.hostNetwork {
		// FIXME
		// force to set hostnetwork
		pod.Spec.HostNetwork = true
	}
}

func getPodFromTemplate(object CommonInterface, spec *v1.PodTemplateSpec, containerPara *ContainerPara) *v1.Pod {

	pod, _ := k8scontroller.GetPodFromTemplate(spec, object, metav1.NewControllerRef(object, object.GroupVersionKind()))
	injectContainerPara(pod, containerPara, object)
	return pod
}
