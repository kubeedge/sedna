package runtime

import (
	"context"
	"fmt"
	"path/filepath"
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
	Mounts []WorkerMount

	Env        map[string]string
	WorkerType string

	// if true, force to use hostNetwork
	HostNetwork bool

	ModelHotUpdate ModelHotUpdate

	RestartPolicy v1.RestartPolicy
}

type ModelHotUpdate struct {
	Enable            bool
	PollPeriodSeconds int64
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

// GenerateWorkerSelector generates the selector of an object for specific worker type
func GenerateWorkerSelector(object CommonInterface, workerType string) (labels.Selector, error) {
	ls := &metav1.LabelSelector{
		// select any type workers
		MatchLabels: generateLabels(object, workerType),
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
			Namespace: object.GetNamespace(),
			Name:      strings.ToLower(name + "-" + workerType),
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

// injectWorkerParam modifies pod in-place
func injectWorkerParam(pod *v1.Pod, workerParam *WorkerParam, object CommonInterface) {
	InjectStorageInitializer(pod, workerParam)

	if workerParam.WorkerType == InferencePodType && workerParam.ModelHotUpdate.Enable {
		injectModelHotUpdateMount(pod, object)
		setModelHotUpdateEnv(workerParam)
	}

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

// CreateEdgeMeshService creates a kubeedge edgemesh service for an object, and returns an edgemesh service URL.
// Since edgemesh can realize Cross-Edge-Cloud communication, the service can be created both on the cloud or edge side.
func CreateEdgeMeshService(kubeClient kubernetes.Interface, object CommonInterface, workerType string, servicePort int32) (string, error) {
	ctx := context.Background()
	name := object.GetName()
	namespace := object.GetNamespace()
	kind := object.GroupVersionKind().Kind
	targetPort := intstr.IntOrString{
		IntVal: servicePort,
	}
	serviceSpec := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      strings.ToLower(name + "-" + workerType),
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(object, object.GroupVersionKind()),
			},
			Labels: generateLabels(object, workerType),
		},
		Spec: v1.ServiceSpec{
			Selector: generateLabels(object, workerType),
			Ports: []v1.ServicePort{
				{
					// TODO: be clean, Port.Name is currently required by edgemesh(v1.8.0).
					// and should be <protocol>-<suffix>
					Name: "tcp-0",

					Protocol:   "TCP",
					Port:       servicePort,
					TargetPort: targetPort,
				},
			},
		},
	}
	service, err := kubeClient.CoreV1().Services(namespace).Create(ctx, serviceSpec, metav1.CreateOptions{})
	if err != nil {
		klog.Warningf("failed to create service for %v %v/%v, err:%s", kind, namespace, name, err)
		return "", err
	}

	klog.V(2).Infof("Service %s is created successfully for %v %v/%v", service.Name, kind, namespace, name)
	return fmt.Sprintf("%s.%s", service.Name, service.Namespace), nil
}

// CreateDeploymentWithTemplate creates and returns a deployment object given a crd object, deployment template
func CreateDeploymentWithTemplate(client kubernetes.Interface, object CommonInterface, spec *appsv1.DeploymentSpec, workerParam *WorkerParam, port int32) (*appsv1.Deployment, error) {
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

func newDeployment(object CommonInterface, spec *appsv1.DeploymentSpec, workerParam *WorkerParam) *appsv1.Deployment {
	nameSpace := object.GetNamespace()
	deploymentName := object.GetName() + "-" + "deployment" + "-" + strings.ToLower(workerParam.WorkerType) + "-"
	matchLabel := make(map[string]string)
	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: deploymentName,
			Namespace:    nameSpace,
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(object, object.GroupVersionKind()),
			},
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: (*spec).Replicas,
			Template: (*spec).Template,
			Selector: &metav1.LabelSelector{
				MatchLabels: matchLabel,
			},
		},
	}
}

// injectDeploymentParam modifies deployment in-place
func injectDeploymentParam(deployment *appsv1.Deployment, workerParam *WorkerParam, object CommonInterface, _port int32) {
	var appLabelKey = "app.sedna.io"
	var appLabelValue = object.GetName() + "-" + workerParam.WorkerType + "-" + "svc"

	// Injection of the storage variables must be done before loading
	// the environment variables!
	if workerParam.Mounts != nil {
		InjectStorageInitializerDeployment(deployment, workerParam)
	}

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

	for k, v := range generateLabels(object, workerParam.WorkerType) {
		deployment.Labels[k] = v
		deployment.Spec.Template.Labels[k] = v
		deployment.Spec.Selector.MatchLabels[k] = v
	}

	// Edgemesh part, useful for service mapping (not necessary!)
	deployment.Labels[appLabelKey] = appLabelValue
	deployment.Spec.Template.Labels[appLabelKey] = appLabelValue
	deployment.Spec.Selector.MatchLabels[appLabelKey] = appLabelValue

	// Env variables injection
	envs := createEnvVars(workerParam.Env)
	for idx := range deployment.Spec.Template.Spec.Containers {
		deployment.Spec.Template.Spec.Containers[idx].Env = append(
			deployment.Spec.Template.Spec.Containers[idx].Env, envs...,
		)
	}
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

// injectModelHotUpdateMount injects volume mounts when worker supports hot update of model
func injectModelHotUpdateMount(pod *v1.Pod, object CommonInterface) {
	hostPathType := v1.HostPathDirectoryOrCreate

	var volumes []v1.Volume
	var volumeMounts []v1.VolumeMount

	modelHotUpdateHostDir, _ := filepath.Split(GetModelHotUpdateConfigFile(object, ModelHotUpdateHostPrefix))
	volumeName := ConvertK8SValidName(ModelHotUpdateVolumeName)
	volumes = append(volumes, v1.Volume{
		Name: volumeName,
		VolumeSource: v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{
				Path: modelHotUpdateHostDir,
				Type: &hostPathType,
			},
		},
	})

	volumeMounts = append(volumeMounts, v1.VolumeMount{
		MountPath: ModelHotUpdateContainerPrefix,
		Name:      volumeName,
	})

	injectVolume(pod, volumes, volumeMounts)
}

func GetModelHotUpdateConfigFile(object CommonInterface, prefix string) string {
	return strings.ToLower(filepath.Join(prefix, object.GetNamespace(), object.GetObjectKind().GroupVersionKind().Kind,
		object.GetName(), ModelHotUpdateConfigFile))
}

// setModelHotUpdateEnv sets envs of model hot update
func setModelHotUpdateEnv(workerParam *WorkerParam) {
	workerParam.Env["MODEL_HOT_UPDATE"] = "true"
	workerParam.Env["MODEL_POLL_PERIOD_SECONDS"] = strconv.FormatInt(workerParam.ModelHotUpdate.PollPeriodSeconds, 10)
	workerParam.Env["MODEL_HOT_UPDATE_CONFIG"] = filepath.Join(ModelHotUpdateContainerPrefix, ModelHotUpdateConfigFile)
}
