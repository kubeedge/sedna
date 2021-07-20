package runtime

import (
	"context"
	"strings"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/intstr"
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
	Mounts []WorkerMount

	Env        map[string]string
	WorkerType string

	// if true, force to use hostNetwork
	HostNetwork bool

	RestartPolicy v1.RestartPolicy
}

// generateLabels generates labels for an object
func generateLabels(object CommonInterface, workerType string) map[string]string {
	kind := object.GroupVersionKind().Kind
	group := object.GroupVersionKind().Group

	keyPrefix := strings.ToLower(kind + "." + group + "/")

	labels := make(map[string]string)
	labels[keyPrefix+"name"] = object.GetName()
	labels[keyPrefix+"uid"] = string(object.GetUID())
	if workerType != "" {
		labels[keyPrefix+"worker-type"] = strings.ToLower(workerType)
	}
	return labels
}

// GenerateSelector generates the selector of an object for worker
func GenerateSelector(object CommonInterface) (labels.Selector, error) {
	ls := &metav1.LabelSelector{
		// select any type workers
		MatchLabels: generateLabels(object, ""),
	}
	return metav1.LabelSelectorAsSelector(ls)
}

// CreateKubernetesService creates a k8s service for an object given ip and port
func CreateKubernetesService(kubeClient kubernetes.Interface, object CommonInterface, workerType string, inputPort int32, inputIP string) (int32, error) {
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
			Labels: generateLabels(object, workerType),
		},
		Spec: v1.ServiceSpec{
			Selector: generateLabels(object, workerType),
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

// injectWorkerParam.Modifies pod in-place
func injectWorkerParam(pod *v1.Pod, workerParam *WorkerParam, object CommonInterface) {
	InjectStorageInitializer(pod, workerParam)

	envs := createEnvVars(workerParam.Env)
	for idx := range pod.Spec.Containers {
		pod.Spec.Containers[idx].Env = append(
			pod.Spec.Containers[idx].Env, envs...,
		)
	}

	// inject our labels
	if pod.Labels == nil {
		pod.Labels = make(map[string]string)
	}

	for k, v := range generateLabels(object, workerParam.WorkerType) {
		pod.Labels[k] = v
	}

	pod.GenerateName = object.GetName() + "-" + strings.ToLower(workerParam.WorkerType) + "-"

	pod.Namespace = object.GetNamespace()

	if workerParam.HostNetwork {
		// FIXME
		// force to set hostnetwork
		pod.Spec.HostNetwork = true
	}

	if pod.Spec.RestartPolicy == "" {
		pod.Spec.RestartPolicy = workerParam.RestartPolicy
	}
}

// CreatePodWithTemplate creates and returns a pod object given a crd object, pod template, and workerParam
func CreatePodWithTemplate(client kubernetes.Interface, object CommonInterface, spec *v1.PodTemplateSpec, workerParam *WorkerParam) (*v1.Pod, error) {
	objectKind := object.GroupVersionKind()
	pod, _ := k8scontroller.GetPodFromTemplate(spec, object, metav1.NewControllerRef(object, objectKind))
	injectWorkerParam(pod, workerParam, object)

	createdPod, err := client.CoreV1().Pods(object.GetNamespace()).Create(context.TODO(), pod, metav1.CreateOptions{})
	objectName := object.GetNamespace() + "/" + object.GetName()
	if err != nil {
		klog.Warningf("failed to create pod(type=%s) for %s %s, err:%s", workerParam.WorkerType, objectKind, objectName, err)
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
