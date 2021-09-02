package globalmanager

import (
	"context"
	"strconv"
	"strings"

	appsv1 "k8s.io/api/apps/v1"
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
	mounts []WorkerMount

	env        map[string]string
	workerType string

	// if true, force to use hostNetwork
	hostNetwork bool

	restartPolicy v1.RestartPolicy
}

// generateLabels generates labels for an object
func generateLabels(object CommonInterface, workerType string) map[string]string {
	kind := object.GroupVersionKind().Kind
	group := object.GroupVersionKind().Group

	keyPrefix := strings.ToLower(kind + "." + group + "/")

	labels := make(map[string]string)
	//labels["app"] = object.GetName() + "-" + workerType + "-" + "svc"
	labels[keyPrefix+"name"] = object.GetName()
	labels[keyPrefix+"uid"] = string(object.GetUID())
	if workerType != "" {
		labels[keyPrefix+"worker-type"] = strings.ToLower(workerType)
	}
	return labels
}

func GenerateEdgeMeshSelector(workerName string) map[string]string {
	selector := make(map[string]string)
	selector["app"] = workerName

	return selector
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
			GenerateName: name + "-" + workerType + "-" + "svc",
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(object, object.GroupVersionKind()),
			},
			Labels: generateLabels(object, workerType),
		},
		Spec: v1.ServiceSpec{
			Selector: GenerateEdgeMeshSelector(name + "-" + workerType + "-" + "svc"),
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

	for k, v := range generateLabels(object, workerParam.workerType) {
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

//by EnfangCui
//CreateEdgeMeshService creates a kubeedge edgemesh service for an object given port
func CreateEdgeMeshService(kubeClient kubernetes.Interface, object CommonInterface, workerType string, inputPort int32) (string, error) {
	ctx := context.Background() //TODO 为什么要使用Background？为什么不用TODO()?
	name := object.GetName()
	namespace := object.GetNamespace()
	kind := object.GroupVersionKind().Kind
	targePort := intstr.IntOrString{
		IntVal: inputPort,
	}

	serviceSpec := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      name + "-" + workerType + "-" + "svc", //by EnfangCui这里直接用Name，没用GenerateName
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(object, object.GroupVersionKind()),
			},
			Labels: generateLabels(object, workerType),
		},
		Spec: v1.ServiceSpec{
			//Selector: generateLabels(object, workerType),
			Selector: GenerateEdgeMeshSelector(name + "-" + workerType + "-" + "svc"),
			Ports: []v1.ServicePort{
				{
					Name:       "http-0",
					Protocol:   "TCP",
					Port:       inputPort,
					TargetPort: targePort,
				},
			},
		},
	}

	service, err := kubeClient.CoreV1().Services(namespace).Create(ctx, serviceSpec, metav1.CreateOptions{})
	if err != nil {
		klog.Warningf("failed to create service for %v %v/%v, err:%s", kind, namespace, name, err)
		return "0", err
	}

	klog.V(2).Infof("Service %s is created successfully for %v %v/%v", service.Name, kind, namespace, name)
	edgeMeshURL := name + "-" + workerType + "-" + "svc" + "." + namespace + ":" + strconv.Itoa(int(inputPort))

	return edgeMeshURL, nil //这返回一个url by EnfangCui
}

//by EnfangCui
func newDeployment(object CommonInterface, spec *appsv1.DeploymentSpec, workerParam *WorkerParam) *appsv1.Deployment {
	nameSpace := object.GetNamespace()
	deploymentName := object.GetName() + "-" + "deployment" + "-" + strings.ToLower(workerParam.workerType) + "-"
	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: deploymentName,
			Namespace:    nameSpace,
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(object, object.GroupVersionKind()),
			},
		},
		Spec: *spec,
	}
}

//by EnfangCui
// injectDeploymentParam modifies deployment in-place
func injectDeploymentParam(deployment *appsv1.Deployment, workerParam *WorkerParam, object CommonInterface, port int32) {
	// inject our labels
	if deployment.Labels == nil {
		deployment.Labels = make(map[string]string)
	}
	if deployment.Spec.Template.Labels == nil {
		deployment.Spec.Template.Labels = make(map[string]string)
	}
	if deployment.Spec.Selector.MatchLabels == nil {
		deployment.Spec.Selector.MatchLabels = make(map[string]string)
	}

	for k, v := range generateLabels(object, workerParam.workerType) {
		deployment.Labels[k] = v
		deployment.Spec.Template.Labels[k] = v
		deployment.Spec.Selector.MatchLabels[k] = v
	}

	// Edgemesh part
	deployment.Labels["app"] = object.GetName() + "-" + workerParam.workerType + "-" + "svc"
	deployment.Spec.Template.Labels["app"] = object.GetName() + "-" + workerParam.workerType + "-" + "svc"
	deployment.Spec.Selector.MatchLabels["app"] = object.GetName() + "-" + workerParam.workerType + "-" + "svc"

	if deployment.Spec.Template.Spec.Containers[0].Ports != nil {
		deployment.Spec.Template.Spec.Containers[0].Ports[0].HostPort = port
		deployment.Spec.Template.Spec.Containers[0].Ports[0].ContainerPort = port
	}

	envs := createEnvVars(workerParam.env)
	for idx := range deployment.Spec.Template.Spec.Containers {
		deployment.Spec.Template.Spec.Containers[idx].Env = append(
			deployment.Spec.Template.Spec.Containers[idx].Env, envs...,
		)
	}

	InjectStorageInitializerDeployment(deployment, workerParam)


}

//by EnfangCui
//CreateDeploymentWithTemplate creates and returns a deployment object given a crd object, deployment template
func CreateDeploymentWithTemplate(client kubernetes.Interface, object CommonInterface, spec *appsv1.DeploymentSpec, workerParam *WorkerParam, port int32) (*appsv1.Deployment, error) {
	klog.Infof("Creating new deployment %v!", object.GetName())

	objectKind := object.GroupVersionKind()
	objectName := object.GetNamespace() + "/" + object.GetName()
	deployment := newDeployment(object, spec, workerParam)

	injectDeploymentParam(deployment, workerParam, object, port)
	createdDeployment, err := client.AppsV1().Deployments(object.GetNamespace()).Create(context.TODO(), deployment, metav1.CreateOptions{})
	if err != nil {
		klog.Warningf("failed to create deployment for %s %s, err:%s", objectKind, objectName, err)
		return nil, err
	}
	klog.V(2).Infof("deployment %s is created successfully for %s %s", createdDeployment.Name, objectKind, objectName)
	return createdDeployment, nil

}

func refreshDeploymentAndService(client kubernetes.Interface, object CommonInterface, deployment *appsv1.Deployment, service *v1.Service) (err error) {
	klog.Info("Launching deploymeny and service update routine")
	objectKind := object.GroupVersionKind()
	objectName := object.GetNamespace() + "/" + object.GetName()

	_, err = client.AppsV1().Deployments(object.GetNamespace()).Patch(context.TODO(), "", "", nil, metav1.PatchOptions{})
	if err != nil {
		klog.Warningf("failed to update deployment for %s %s, err:%s", objectKind, objectName, err)
		return err
	}

	_, err = client.CoreV1().Services(object.GetNamespace()).Update(context.TODO(), service, metav1.UpdateOptions{})
	if err != nil {
		klog.Warningf("failed to update deployment for %s %s, err:%s", objectKind, objectName, err)
		return err
	}

	return nil

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
