package globalmanager

import (
	"context"
	"strings"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	k8scontroller "k8s.io/kubernetes/pkg/controller"
)

type WorkerMount struct {
	Name string
	// the url to be mounted
	URL *MountURL

	// for some cases, there are more than one url to be mounted
	URLs []MountURL

	// envName indicates the environment key of the mounts injected to the worker
	EnvName string
}

// WorkerParam describes the system-defined parameters of worker
type WorkerParam struct {
	mounts []WorkerMount

	env        map[string]string
	workerType string

	// if true, force to use hostNetwork
	hostNetwork bool

	restartPolicy v1.RestartPolicy
}

// injectWorkerParam modifies pod in-place
func injectWorkerParam(pod *v1.Pod, workerParam *WorkerParam, object CommonInterface) {
	InjectStorageInitializer(pod, workerParam)

	envs := createEnvVars(workerParam.env)
	for idx := range pod.Spec.Containers {
		pod.Spec.Containers[idx].Env = append(
			pod.Spec.Containers[idx].Env, envs...,
		)
	}

	// inject our labels
	if pod.Labels == nil {
		pod.Labels = make(map[string]string)
	}

	for k, v := range GenerateLabels(object) {
		pod.Labels[k] = v
	}

	pod.GenerateName = object.GetName() + "-" + strings.ToLower(workerParam.workerType) + "-"

	pod.Namespace = object.GetNamespace()

	if workerParam.hostNetwork {
		// FIXME
		// force to set hostnetwork
		pod.Spec.HostNetwork = true
	}

	if pod.Spec.RestartPolicy == "" {
		pod.Spec.RestartPolicy = workerParam.restartPolicy
	}
}

// createPodWithTemplate creates and returns a pod object given a crd object, pod template, and workerParam
func createPodWithTemplate(client kubernetes.Interface, object CommonInterface, spec *v1.PodTemplateSpec, workerParam *WorkerParam) (*v1.Pod, error) {
	objectKind := object.GroupVersionKind()
	pod, _ := k8scontroller.GetPodFromTemplate(spec, object, metav1.NewControllerRef(object, objectKind))
	injectWorkerParam(pod, workerParam, object)

	createdPod, err := client.CoreV1().Pods(object.GetNamespace()).Create(context.TODO(), pod, metav1.CreateOptions{})
	objectName := object.GetNamespace() + "/" + object.GetName()
	if err != nil {
		klog.Warningf("failed to create pod(type=%s) for %s %s, err:%s", workerParam.workerType, objectKind, objectName, err)
		return nil, err
	}
	klog.V(2).Infof("pod %s is created successfully for %s %s", createdPod.Name, objectKind, objectName)
	return createdPod, nil
}

// createEnvVars creates EnvMap for container
// include EnvName and EnvValue map for stage of creating a pod
func createEnvVars(envMap map[string]string) []v1.EnvVar {
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
