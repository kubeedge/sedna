/*


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

package controllers

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/go-logr/logr"
	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	utilrand "k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/discovery/cached/memory"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/restmapper"
	watchtools "k8s.io/client-go/tools/watch"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	aiv1alpha1 "github.com/kubeedge/sedna/components/modeloperator/api/v1alpha1"
)

// InferenceModelReconciler reconciles a InferenceModel object
type InferenceModelReconciler struct {
	client.Client
	Log    logr.Logger
	Scheme *runtime.Scheme
	Config *rest.Config
}

// +kubebuilder:rbac:groups=sedna.io,resources=inferencemodels,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=sedna.io,resources=inferencemodels/status,verbs=get;update;patch

// Reconcile does the reconcilation between desired state and actual state
func (r *InferenceModelReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := r.Log.WithValues("inferencemodel", req.NamespacedName)
	log.Info("reconciling inferencemodel")

	var model aiv1alpha1.InferenceModel
	if err := r.Get(ctx, req.NamespacedName, &model); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	if model.Spec.DeployToLayer == "cloud" {
		return r.reconcileCloud(ctx, model, log)
	} else if model.Spec.DeployToLayer == "edge" {
		return r.reconcileEdge(ctx, model, log)
	} else {
		log.Error(fmt.Errorf("DeployToLayer not supported"), "DeployToLayer is "+model.Spec.DeployToLayer)
		return ctrl.Result{}, nil
	}
}

// SetupWithManager register reconciler with controller manager
func (r *InferenceModelReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&aiv1alpha1.InferenceModel{}).
		Owns(&corev1.Pod{}).
		Owns(&corev1.Service{}).
		Owns(&appsv1.Deployment{}).
		Complete(r)
}

func (r *InferenceModelReconciler) reconcileCloud(
	ctx context.Context,
	model aiv1alpha1.InferenceModel,
	log logr.Logger) (ctrl.Result, error) {
	return ctrl.Result{}, nil
}

func (r *InferenceModelReconciler) reconcileEdge(
	ctx context.Context,
	model aiv1alpha1.InferenceModel,
	log logr.Logger) (ctrl.Result, error) {
	if model.Status.ServingVersion == model.Spec.TargetVersion {
		var pod corev1.Pod
		key := types.NamespacedName{
			Namespace: model.Namespace,
			Name:      fmt.Sprintf("%s-%s", "inference", model.Name),
		}
		if err := r.Get(ctx, key, &pod); err == nil {
			for _, c := range pod.Status.Conditions {
				if c.Type == corev1.PodReady && c.Status == corev1.ConditionTrue {
					log.Info("Skip reconileEdge, the model is already serving the target version: " + model.Spec.TargetVersion)
					return ctrl.Result{}, nil
				}
			}
		}
	}

	err := r.downloadModelAndWait(ctx, model, log)
	if err != nil {
		log.Info(fmt.Sprintf("Error in downloading model: %s.", err.Error()))
		return ctrl.Result{Requeue: true}, nil
	}

	pod, err := r.desiredPod(model)
	if err != nil {
		return ctrl.Result{}, err
	}

	applyOpts := []client.PatchOption{client.ForceOwnership, client.FieldOwner("modeloperator")}
	err = r.Patch(ctx, &pod, client.Apply, applyOpts...)
	if err != nil {
		log.Info(fmt.Sprintf("Error in patching model: %s.", err.Error()))
		return ctrl.Result{Requeue: true}, nil
	}

	// Wait for Pod running
	podinfo, _ := r.getResourceInfo(pod.GroupVersionKind(), pod.Name, pod.Namespace)
	_, err = r.waitForResource(ctx, podinfo, 5*time.Minute, checkPodCondition)
	if err != nil {
		log.Info(fmt.Sprintf("Error in pod: %s.", err.Error()))
		return ctrl.Result{}, err
	}

	// Pod is ready to serve traffic
	model.Status.ServingVersion = model.Spec.TargetVersion
	r.Status().Update(ctx, &model)

	// Wait for pod status to sync into local cache
	time.Sleep(10 * time.Second)

	log.Info("reconciled inferencemodel on the edge")
	return ctrl.Result{}, nil
}

func (r *InferenceModelReconciler) desiredPod(model aiv1alpha1.InferenceModel) (corev1.Pod, error) {
	pod := corev1.Pod{
		TypeMeta: metav1.TypeMeta{APIVersion: corev1.SchemeGroupVersion.String(), Kind: "Pod"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-%s", "inference", model.Name),
			Namespace: model.Namespace,
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name:  fmt.Sprintf("%s-%s", "inference", model.Name),
					Image: "tensorflow/serving:latest",
					VolumeMounts: []corev1.VolumeMount{
						{
							Name:      "modelfiles-" + model.Name,
							MountPath: "/models/" + model.Name,
						},
					},
					Ports: []corev1.ContainerPort{
						{
							Protocol:      corev1.ProtocolTCP,
							ContainerPort: 8501,
							HostPort:      8501,
						},
					},
					Args: []string{
						fmt.Sprintf("--model_config_file=/models/%s/models.config", model.Name),
						"--model_config_file_poll_wait_seconds=60",
					},
				},
			},
			NodeName: model.Spec.NodeName,
			Volumes: []corev1.Volume{
				{
					Name: "modelfiles-" + model.Name,
					VolumeSource: corev1.VolumeSource{
						HostPath: &corev1.HostPathVolumeSource{
							Path: "/var/lib/sedna/models/" + model.Name,
						},
					},
				},
			},
		},
	}

	if err := ctrl.SetControllerReference(&model, &pod, r.Scheme); err != nil {
		return pod, err
	}

	return pod, nil
}

func (r *InferenceModelReconciler) downloadModelAndWait(ctx context.Context, model aiv1alpha1.InferenceModel, log logr.Logger) error {
	defaultMode := int32(0744)
	one := int32(1)

	indx := searchModelFile(model.Spec.TargetVersion, model)
	if indx == -1 {
		return fmt.Errorf("Target version cannot be found in the manifest: %s", model.Spec.TargetVersion)
	}

	modelfile := model.Spec.Manifest[indx]

	downloadJob := batchv1.Job{
		TypeMeta: metav1.TypeMeta{APIVersion: batchv1.SchemeGroupVersion.String(), Kind: "Job"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-%s-%s", model.Name, "downloadmodel", utilrand.String(5)),
			Namespace: model.Namespace,
		},
		Spec: batchv1.JobSpec{
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:  "download-model",
							Image: "busybox:1.28",
							VolumeMounts: []corev1.VolumeMount{
								{
									Name:      "sednahome",
									MountPath: "/var/lib/sedna",
								},
								{
									Name:      "scripts",
									MountPath: "/scripts",
								},
							},
							Command: []string{"bin/sh"},
							Args: []string{
								"-c",
								fmt.Sprintf(
									"/scripts/downloadModelFile.sh %s %s %s %s",
									model.Spec.ModelName,
									modelfile.Version,
									modelfile.DownloadURL,
									modelfile.Sha256sum),
							},
						},
					},
					NodeName: model.Spec.NodeName,
					Volumes: []corev1.Volume{
						{
							Name: "sednahome",
							VolumeSource: corev1.VolumeSource{
								HostPath: &corev1.HostPathVolumeSource{
									Path: "/var/lib/sedna",
								},
							},
						},
						{
							Name: "scripts",
							VolumeSource: corev1.VolumeSource{
								ConfigMap: &corev1.ConfigMapVolumeSource{
									LocalObjectReference: corev1.LocalObjectReference{
										Name: "sedna-downloadmodelfile",
									},
									DefaultMode: &defaultMode,
								},
							},
						},
					},
					RestartPolicy: corev1.RestartPolicyNever,
				},
			},
			BackoffLimit: &one,
		},
	}

	if err := ctrl.SetControllerReference(&model, &downloadJob, r.Scheme); err != nil {
		return err
	}

	var obj runtime.Object
	var thejob *unstructured.Unstructured
	var jobinfo *resource.Info

	applyOpts := []client.PatchOption{client.ForceOwnership, client.FieldOwner("modeloperator")}
	err := r.Patch(ctx, &downloadJob, client.Apply, applyOpts...)
	if err != nil {
		goto cleanup
	}

	jobinfo, err = r.getResourceInfo(downloadJob.GroupVersionKind(), downloadJob.Name, downloadJob.Namespace)
	if err != nil {
		goto cleanup
	}

	obj, err = r.waitForResource(ctx, jobinfo, 10*time.Minute, checkJobCondition)
	if err != nil {
		goto cleanup
	}

	thejob = obj.(*unstructured.Unstructured)
	if checkConditionHelper(thejob, string(batchv1.JobFailed), string(corev1.ConditionTrue)) {
		err = fmt.Errorf("Download job failed: %s/%s", thejob.GetNamespace(), thejob.GetName())
	}

cleanup:
	zero := int64(0)
	deletepolicy := metav1.DeletePropagationForeground
	deleteOpts := []client.DeleteOption{&client.DeleteOptions{
		GracePeriodSeconds: &zero,
		PropagationPolicy:  &deletepolicy,
	}}
	r.Delete(ctx, &downloadJob, deleteOpts...)
	return err
}

func (r *InferenceModelReconciler) waitForResource(
	ctx context.Context, info *resource.Info, timeout time.Duration,
	checkCondition checkCondition) (runtime.Object, error) {
	dynamicClient, _ := dynamic.NewForConfig(r.Config)
	endtime := time.Now().Add(timeout)
	nameSelector := fields.OneTermEqualSelector("metadata.name", info.Name).String()
	for {
		objlist, err := dynamicClient.Resource(info.Mapping.Resource).Namespace(info.Namespace).List(ctx, metav1.ListOptions{FieldSelector: nameSelector})
		if err != nil {
			return nil, err
		}

		if len(objlist.Items) == 0 {
			return nil, fmt.Errorf("Cannot find resource %s: %s/%s", info.Mapping.Resource.Resource, info.Namespace, info.Name)
		}

		obj := &objlist.Items[0]
		if isConditionMet := checkCondition(obj); isConditionMet {
			return obj, nil
		}

		curTimeout := endtime.Sub(time.Now())
		if curTimeout < 0 {
			return obj, fmt.Errorf("Timed out waiting for %s: %s/%s", info.Mapping.Resource, info.Namespace, info.Name)
		}

		resourceVersion := objlist.GetResourceVersion()

		objwatch, err := dynamicClient.Resource(info.Mapping.Resource).Namespace(info.Namespace).Watch(ctx, metav1.ListOptions{
			FieldSelector:   nameSelector,
			ResourceVersion: resourceVersion,
		})

		if err != nil {
			return obj, err
		}

		ctx, cancel := watchtools.ContextWithOptionalTimeout(ctx, curTimeout)
		event, err := watchtools.UntilWithoutRetry(ctx, objwatch, func(event watch.Event) (bool, error) {
			if event.Type == watch.Error {
				err := apierrors.FromObject(event.Object)
				r.Log.Error(err, "error: An error occurred while waiting for the condition to be satisfied")
				return false, nil
			}
			if event.Type == watch.Deleted {
				return false, nil
			}
			obj := event.Object.(*unstructured.Unstructured)
			return checkCondition(obj), nil
		})
		cancel()
		switch {
		case err == nil:
			return event.Object, nil
		case err == watchtools.ErrWatchClosed:
			continue
		case err == wait.ErrWaitTimeout:
			return obj, fmt.Errorf("Timed out waiting for %s: %s/%s", info.Mapping.Resource, info.Namespace, info.Name)
		default:
			return obj, err
		}

	}
}

func (r *InferenceModelReconciler) getResourceInfoFromPod(pod corev1.Pod) *resource.Info {
	dc, _ := discovery.NewDiscoveryClientForConfig(r.Config)
	mapper := restmapper.NewDeferredDiscoveryRESTMapper(memory.NewMemCacheClient(dc))
	mapping, _ := mapper.RESTMapping(pod.GroupVersionKind().GroupKind(), pod.APIVersion)
	return &resource.Info{
		Mapping:   mapping,
		Name:      pod.Name,
		Namespace: pod.Namespace,
	}
}

func (r *InferenceModelReconciler) getResourceInfo(gvk schema.GroupVersionKind, name, namespace string) (*resource.Info, error) {
	dc, err := discovery.NewDiscoveryClientForConfig(r.Config)
	if err != nil {
		return nil, err
	}

	mapper := restmapper.NewDeferredDiscoveryRESTMapper(memory.NewMemCacheClient(dc))
	mapping, err := mapper.RESTMapping(gvk.GroupKind(), gvk.Version)

	if err != nil {
		return nil, err
	}

	return &resource.Info{
		Mapping:   mapping,
		Name:      name,
		Namespace: namespace,
	}, nil
}

func searchModelFile(version string, model aiv1alpha1.InferenceModel) int {
	for i, f := range model.Spec.Manifest {
		if f.Version == version {
			return i
		}
	}
	return -1
}

type checkCondition func(obj *unstructured.Unstructured) bool

func checkJobCondition(obj *unstructured.Unstructured) bool {
	return checkConditionHelper(obj, string(batchv1.JobComplete), string(corev1.ConditionTrue)) ||
		checkConditionHelper(obj, string(batchv1.JobFailed), string(corev1.ConditionTrue))
}

func checkPodCondition(obj *unstructured.Unstructured) bool {
	return checkConditionHelper(obj, string(corev1.PodReady), string(corev1.ConditionTrue))
}

func checkConditionHelper(obj *unstructured.Unstructured, conditionName, conditionStatus string) bool {
	conditions, found, err := unstructured.NestedSlice(obj.Object, "status", "conditions")
	if err != nil {
		return false
	}
	if !found {
		return false
	}
	for _, conditionUncast := range conditions {
		condition := conditionUncast.(map[string]interface{})
		name, found, err := unstructured.NestedString(condition, "type")
		if !found || err != nil || !strings.EqualFold(name, conditionName) {
			continue
		}
		status, found, err := unstructured.NestedString(condition, "status")
		if !found || err != nil {
			continue
		}
		return strings.EqualFold(status, conditionStatus)
	}

	return false
}
