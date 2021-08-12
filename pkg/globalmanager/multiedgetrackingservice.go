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
	"reflect"
	"strconv"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilrand "k8s.io/apimachinery/pkg/util/rand"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	kubeinformers "k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	appslisters "k8s.io/client-go/listers/apps/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	k8scontroller "k8s.io/kubernetes/pkg/controller"

	sednav1 "github.com/kubeedge/sedna/pkg/apis/sedna/v1alpha1"
	clientset "github.com/kubeedge/sedna/pkg/client/clientset/versioned"
	sednaclientset "github.com/kubeedge/sedna/pkg/client/clientset/versioned/typed/sedna/v1alpha1"
	informers "github.com/kubeedge/sedna/pkg/client/informers/externalversions"
	sednav1listers "github.com/kubeedge/sedna/pkg/client/listers/sedna/v1alpha1"
	"github.com/kubeedge/sedna/pkg/globalmanager/config"
	messageContext "github.com/kubeedge/sedna/pkg/globalmanager/messagelayer/ws"
	"github.com/kubeedge/sedna/pkg/globalmanager/utils"
)

const (
	MultiObjectTrackingWorker = "l1-edge"
	FEWorker                  = "l2-edge"
	ReIDWoker                 = "cloud"
	reIDPort                  = 5000
	FEPort                    = 6000
)

const (
	// Name is this controller name
	Name = "MultiEdgeTracking"

	// KindName is the kind name of CR this controller controls
	KindName = "MultiEdgeTrackingService"
)

// MultiEdgeTrackingServicerKind contains the schema.GroupVersionKind for this controller type.
var MultiEdgeTrackingServiceKind = sednav1.SchemeGroupVersion.WithKind("MultiEdgeTrackingService")

// MultiEdgeTrackingServiceController ensures that all MultiEdgeTrackingService objects
// have corresponding pods to run their configured workload.
type MultiEdgeTrackingServiceController struct {
	kubeClient kubernetes.Interface
	client     sednaclientset.SednaV1alpha1Interface

	// podStoreSynced returns true if the pod store has been synced at least once.
	podStoreSynced cache.InformerSynced
	// A store of pods
	podStore corelisters.PodLister

	// serviceStoreSynced returns true if the jointinferenceservice store has been synced at least once.
	serviceStoreSynced cache.InformerSynced
	// A store of service
	serviceLister sednav1listers.MultiEdgeTrackingServiceLister

	// deploymentsSynced returns true if the deployment store has been synced at least once.
	deploymentsSynced cache.InformerSynced
	// A store of deployment
	deploymentsLister appslisters.DeploymentLister

	enqueueDeployment func(deployment *appsv1.Deployment)

	// MultiEdgeTrackingService that need to be updated
	queue workqueue.RateLimitingInterface

	recorder record.EventRecorder

	cfg *config.ControllerConfig

	sendToEdgeFunc func(nodeName string, eventType watch.EventType, obj interface{}) error
}

// Start starts the main goroutine responsible for watching and syncing services.
func (mc *MultiEdgeTrackingServiceController) Start() error {
	workers := 1
	stopCh := messageContext.Done()

	go func() {
		defer utilruntime.HandleCrash()
		defer mc.queue.ShutDown()
		klog.Infof("Starting multi-edge tracking (MOT) service controller")
		defer klog.Infof("Shutting down multi-edge tracking (MOT) service controller")

		if !cache.WaitForNamedCacheSync("multiedgetrackingservice", stopCh, mc.podStoreSynced, mc.serviceStoreSynced) {
			klog.Errorf("Error in cache synchronization for MOT service!")

			return
		}

		klog.Infof("Starting MOT service workers...")
		for i := 0; i < workers; i++ {
			go wait.Until(mc.worker, time.Second, stopCh)
		}

		<-stopCh
	}()
	return nil
}

/*

DEPLOYMENT HOOKS

*/

func (mc *MultiEdgeTrackingServiceController) enqueueByDeployment(deployment *appsv1.Deployment) {
	controllerRef := metav1.GetControllerOf(deployment)

	klog.Info("Deployment enqueued %w", deployment.Kind)

	if controllerRef == nil {
		return
	}

	if controllerRef.Kind != MultiEdgeTrackingServiceKind.Kind {
		return
	}

	service, err := mc.serviceLister.MultiEdgeTrackingServices(deployment.Namespace).Get(controllerRef.Name)
	if err != nil {
		return
	}

	if service.UID != controllerRef.UID {
		return
	}

	mc.enqueueController(service, true)
}

func (mc *MultiEdgeTrackingServiceController) addDeployment(obj interface{}) {
	deployment := obj.(*appsv1.Deployment)
	mc.enqueueByDeployment(deployment)
}

//deleteDeployment enqueues the ObjectSearchService obj When a deleteDeployment is deleted
func (mc *MultiEdgeTrackingServiceController) deleteDeployment(obj interface{}) {
	deployment, ok := obj.(*appsv1.Deployment)

	// comment from https://github.com/kubernetes/kubernetes/blob/master/pkg/controller/deployment/deployment_controller.go
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			klog.Warningf("couldn't get object from tombstone %+v", obj)
			return
		}
		deployment, ok = tombstone.Obj.(*appsv1.Deployment)
		if !ok {
			klog.Warningf("tombstone contained object that is not a Deployment %+v", obj)
			return
		}
	}
	mc.enqueueByDeployment(deployment)
}

// When a deployment is updated, figure out what object search service manage it and wake them up.
func (mc *MultiEdgeTrackingServiceController) updateDeployment(old, cur interface{}) {
	oldD := old.(*appsv1.Deployment)
	curD := cur.(*appsv1.Deployment)
	// no deployment update, no queue
	if curD.ResourceVersion == oldD.ResourceVersion {
		return
	}

	mc.addDeployment(curD)
}

// obj could be an *sednav1.ObjectSearchService, or a DeletionFinalStateUnknown marker item,
// immediate tells the controller to update the status right away, and should
// happen ONLY when there was a successful pod run.
func (mc *MultiEdgeTrackingServiceController) enqueueController(obj interface{}, immediate bool) {
	key, err := k8scontroller.KeyFunc(obj)
	if err != nil {
		klog.Warningf("Couldn't get key for object %+v: %v", obj, err)
		return
	}

	backoff := time.Duration(0)
	if !immediate {
		backoff = getBackoff(mc.queue, key)
	}
	mc.queue.AddAfter(key, backoff)
}

// enqueueByPod enqueues the jointInferenceService object of the specified pod.
func (mc *MultiEdgeTrackingServiceController) enqueueByPod(pod *v1.Pod, immediate bool) {
	controllerRef := metav1.GetControllerOf(pod)

	if controllerRef == nil {
		return
	}

	if controllerRef.Kind != MultiEdgeTrackingServiceKind.Kind {
		return
	}

	service, err := mc.serviceLister.MultiEdgeTrackingServices(pod.Namespace).Get(controllerRef.Name)
	if err != nil {
		return
	}

	if service.UID != controllerRef.UID {
		return
	}

	mc.enqueueController(service, immediate)
}

// When a pod is created, enqueue the controller that manages it and update it's expectations.
func (mc *MultiEdgeTrackingServiceController) addPod(obj interface{}) {
	pod := obj.(*v1.Pod)
	if pod.DeletionTimestamp != nil {
		// on a restart of the controller, it's possible a new pod shows up in a state that
		// is already pending deletion. Prevent the pod from being a creation observation.
		mc.deletePod(pod)
		return
	}

	// backoff to queue when PodFailed
	immediate := pod.Status.Phase != v1.PodFailed

	mc.enqueueByPod(pod, immediate)
}

// When a pod is updated, figure out what joint inference service manage it and wake them up.
func (mc *MultiEdgeTrackingServiceController) updatePod(old, cur interface{}) {
	curPod := cur.(*v1.Pod)
	oldPod := old.(*v1.Pod)

	// no pod update, no queue
	if curPod.ResourceVersion == oldPod.ResourceVersion {
		return
	}

	mc.addPod(curPod)
}

// deletePod enqueues the jointinferenceservice obj When a pod is deleted
func (mc *MultiEdgeTrackingServiceController) deletePod(obj interface{}) {
	pod, ok := obj.(*v1.Pod)

	// comment from https://github.com/kubernetes/kubernetes/blob/master/pkg/controller/job/job_controller.go

	// When a delete is dropped, the relist will notice a pod in the store not
	// in the list, leading to the insertion of a tombstone object which contains
	// the deleted key/value. Note that this value might be stale. If the pod
	// changed labels the new jointinferenceservice will not be woken up till the periodic resync.
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			klog.Warningf("couldn't get object from tombstone %+v", obj)
			return
		}
		pod, ok = tombstone.Obj.(*v1.Pod)
		if !ok {
			klog.Warningf("tombstone contained object that is not a pod %+v", obj)
			return
		}
	}
	mc.enqueueByPod(pod, true)
}

// worker runs a worker thread that just dequeues items, processes them, and marks them done.
// It enforces that the sync is never invoked concurrently with the same key.
func (mc *MultiEdgeTrackingServiceController) worker() {
	for mc.processNextWorkItem() {
	}
}

func (mc *MultiEdgeTrackingServiceController) processNextWorkItem() bool {
	key, quit := mc.queue.Get()
	if quit {
		return false
	}
	defer mc.queue.Done(key)

	forget, err := mc.sync(key.(string))
	if err == nil {
		if forget {
			mc.queue.Forget(key)
		}
		return true
	}

	klog.Warningf("Error syncing MOT service: %v", err)
	mc.queue.AddRateLimited(key)

	return true
}

// sync will sync the MultiEdgeTrackingService with the given key.
// This function is not meant to be invoked concurrently with the same key.
func (mc *MultiEdgeTrackingServiceController) sync(key string) (bool, error) {
	startTime := time.Now()
	defer func() {
		klog.V(4).Infof("Finished syncing MOT service %q (%v)", key, time.Since(startTime))
	}()

	ns, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return false, err
	}
	if len(ns) == 0 || len(name) == 0 {
		return false, fmt.Errorf("Invalid MOT service key %q: either namespace or name is missing", key)
	}
	sharedMultiEdgeTrackingService, err := mc.serviceLister.MultiEdgeTrackingServices(ns).Get(name)
	if err != nil {
		if errors.IsNotFound(err) {
			klog.V(4).Infof("MultiEdgeTrackingService has been deleted: %v", key)
			return true, nil
		}
		return false, err
	}

	MultiEdgeTrackingService := *sharedMultiEdgeTrackingService

	// if jointinferenceservice was finished previously, we don't want to redo the termination
	if isMultiEdgeTrackingServiceFinished(&MultiEdgeTrackingService) {
		return true, nil
	}

	// set kind for jointinferenceservice in case that the kind is None
	// more details at https://github.com/kubernetes/kubernetes/issues/3030
	MultiEdgeTrackingService.SetGroupVersionKind(MultiEdgeTrackingServiceKind)

	selector, _ := GenerateSelector(&MultiEdgeTrackingService)
	pods, err := mc.podStore.Pods(MultiEdgeTrackingService.Namespace).List(selector)
	deployment, err := mc.deploymentsLister.Deployments(MultiEdgeTrackingService.Namespace).List(selector)

	if err != nil {
		return false, err
	}

	klog.V(4).Infof("list MOT service %v/%v, %v pods: %v", MultiEdgeTrackingService.Namespace, MultiEdgeTrackingService.Name, len(pods), pods)

	latestConditionLen := len(MultiEdgeTrackingService.Status.Conditions)

	activePods := calcActivePodCount(pods)
	activeDeployments := calcActiveDeploymentCount(deployment)

	var failedPods, failedDeployment int32 = 0, 0
	// neededCounts requires that N pods should be created successfully
	// N is given by the sum of (# edge pods) + (cloud pod)
	var neededPodCounts int32 = *sharedMultiEdgeTrackingService.Spec.MultiObjectTrackingDeploy.Spec.Replicas +
		*sharedMultiEdgeTrackingService.Spec.FEDeploy.Spec.Replicas +
		*sharedMultiEdgeTrackingService.Spec.ReIDDeploy.Spec.Replicas

	var neededDeploymentCounts int32 = int32(reflect.TypeOf(sednav1.MultiEdgeTrackingServiceSpec{}).NumField())

	// first start
	if MultiEdgeTrackingService.Status.StartTime == nil {
		now := metav1.Now()
		MultiEdgeTrackingService.Status.StartTime = &now
	} else {
		failedPods = neededPodCounts - activePods
		failedDeployment = neededDeploymentCounts - activeDeployments
	}

	var manageServiceErr error
	serviceFailed := false

	var latestConditionType sednav1.MultiEdgeTrackingServiceConditionType = ""

	// get the latest condition type
	// based on that condition updated is appended, not inserted.
	jobConditions := MultiEdgeTrackingService.Status.Conditions
	if len(jobConditions) > 0 {
		latestConditionType = (jobConditions)[len(jobConditions)-1].Type
	}

	var newCondtionType sednav1.MultiEdgeTrackingServiceConditionType
	var reason string
	var message string

	if failedPods > 0 || failedDeployment > 0 {
		serviceFailed = true
		// TODO: Split code to handle deployment failure separately
		// TODO: get the failed worker, and knows that which worker fails, edge inference worker or cloud inference worker
		reason = "workerFailed"
		message = "the worker of MultiEdgeTrackingService failed"
		newCondtionType = sednav1.MultiEdgeTrackingServiceCondFailed
		mc.recorder.Event(&MultiEdgeTrackingService, v1.EventTypeWarning, reason, message)
	} else {
		if len(pods) == 0 {
			activePods, activeDeployments, manageServiceErr = mc.createWorkers(&MultiEdgeTrackingService)
		}
		if manageServiceErr != nil {
			serviceFailed = true
			message = error.Error(manageServiceErr)
			newCondtionType = sednav1.MultiEdgeTrackingServiceCondFailed
			failedPods = neededPodCounts - activePods
			failedDeployment = neededDeploymentCounts - activeDeployments
		} else {
			// TODO: handle the case that the pod phase is PodSucceeded
			newCondtionType = sednav1.MultiEdgeTrackingServiceCondRunning
		}
	}

	//
	if newCondtionType != latestConditionType {
		MultiEdgeTrackingService.Status.Conditions = append(MultiEdgeTrackingService.Status.Conditions, NewMultiEdgeTrackingServiceCondition(newCondtionType, reason, message))
	}
	forget := false

	// no need to update the MultiEdgeTrackingService if the status hasn't changed since last time
	if MultiEdgeTrackingService.Status.Active != activePods || MultiEdgeTrackingService.Status.Failed != failedPods || len(MultiEdgeTrackingService.Status.Conditions) != latestConditionLen {
		MultiEdgeTrackingService.Status.Active = activePods
		MultiEdgeTrackingService.Status.Failed = failedPods

		if err := mc.updateStatus(&MultiEdgeTrackingService); err != nil {
			return forget, err
		}

		if serviceFailed && !isMultiEdgeTrackingServiceFinished(&MultiEdgeTrackingService) {
			// returning an error will re-enqueue MultiEdgeTrackingService after the backoff period
			return forget, fmt.Errorf("failed pod(s) detected for MOT service key %q", key)
		}

		forget = true
	}

	return forget, manageServiceErr
}

// NewJointInferenceServiceCondition creates a new joint condition
func NewMultiEdgeTrackingServiceCondition(conditionType sednav1.MultiEdgeTrackingServiceConditionType, reason, message string) sednav1.MultiEdgeTrackingServiceCondition {
	return sednav1.MultiEdgeTrackingServiceCondition{
		Type:               conditionType,
		Status:             v1.ConditionTrue,
		LastHeartbeatTime:  metav1.Now(),
		LastTransitionTime: metav1.Now(),
		Reason:             reason,
		Message:            message,
	}
}

func (mc *MultiEdgeTrackingServiceController) updateStatus(MultiEdgeTrackingService *sednav1.MultiEdgeTrackingService) error {
	serviceClient := mc.client.MultiEdgeTrackingServices(MultiEdgeTrackingService.Namespace)
	var err error
	for i := 0; i <= ResourceUpdateRetries; i = i + 1 {
		var newMultiEdgeTrackingService *sednav1.MultiEdgeTrackingService
		newMultiEdgeTrackingService, err = serviceClient.Get(context.TODO(), MultiEdgeTrackingService.Name, metav1.GetOptions{})
		if err != nil {
			break
		}
		newMultiEdgeTrackingService.Status = MultiEdgeTrackingService.Status
		if _, err = serviceClient.UpdateStatus(context.TODO(), newMultiEdgeTrackingService, metav1.UpdateOptions{}); err == nil {
			break
		}
	}
	return nil
}

func isMultiEdgeTrackingServiceFinished(j *sednav1.MultiEdgeTrackingService) bool {
	for _, c := range j.Status.Conditions {
		if (c.Type == sednav1.MultiEdgeTrackingServiceCondFailed) && c.Status == v1.ConditionTrue {
			return true
		}
	}
	return false
}

func (mc *MultiEdgeTrackingServiceController) createWorkers(service *sednav1.MultiEdgeTrackingService) (activePods int32, activeDeployments int32, err error) {
	activePods = 0
	activeDeployments = 0

	var workerParam WorkerParam
	workerParam.workerType = ReIDWoker

	// STEP 1 - Create ReID deployment AND related pods (as part of the deployment creation)
	_, err = CreateDeploymentWithTemplate(mc.kubeClient, service, &service.Spec.ReIDDeploy.Spec, &workerParam, reIDPort)
	if err != nil {
		return activePods, activeDeployments, fmt.Errorf("failed to create reid workers deployment: %w", err)
	}
	activeDeployments++

	//STEP 2 - Create edgemesh service for ReID
	reidServiceURL, err := CreateEdgeMeshService(mc.kubeClient, service, ReIDWoker, reIDPort)
	if err != nil {
		return activePods, activeDeployments, fmt.Errorf("failed to create edgemesh service: %w", err)
	}

	activePods++

	// ---- //
	// STEP 0 - Create parameters that will be created by the deployment
	workerParam.workerType = FEWorker
	workerParam.env = map[string]string{
		"NAMESPACE":    service.Namespace,
		"SERVICE_NAME": service.Name,
		"WORKER_NAME":  "feworker-" + utilrand.String(5),

		"REID_MODEL_BIND_IP":   reidServiceURL,
		"REID_MODEL_BIND_PORT": strconv.Itoa(int(reIDPort)),

		"LC_SERVER": mc.cfg.LC.Server,
	}
	workerParam.hostNetwork = true

	// STEP 3 - Create FE deployment AND related pods (as part of the deployment creation)
	_, err = CreateDeploymentWithTemplate(mc.kubeClient, service, &service.Spec.FEDeploy.Spec, &workerParam, FEPort)
	if err != nil {
		return activePods, activeDeployments, fmt.Errorf("failed to create reid workers deployment: %w", err)
	}
	activeDeployments++

	//STEP 4 - Create edgemesh service for FE
	FEServiceURL, err := CreateEdgeMeshService(mc.kubeClient, service, FEWorker, FEPort)
	if err != nil {
		return activePods, activeDeployments, fmt.Errorf("failed to create edgemesh service: %w", err)
	}

	activePods++

	// ---- //

	// STEP 0 - Create parameters that will be created by the deployment
	workerParam.workerType = MultiObjectTrackingWorker
	workerParam.env = map[string]string{
		"NAMESPACE":    service.Namespace,
		"SERVICE_NAME": service.Name,
		"WORKER_NAME":  "motworker-" + utilrand.String(5),

		"FE_MODEL_BIND_IP":   FEServiceURL,
		"FE_MODEL_BIND_PORT": strconv.Itoa(int(FEPort)),

		"LC_SERVER": mc.cfg.LC.Server,
	}
	workerParam.hostNetwork = true

	// STEP 5 - Create OD deployment AND related pods (as part of the deployment creation)
	_, err = CreateDeploymentWithTemplate(mc.kubeClient, service, &service.Spec.MultiObjectTrackingDeploy.Spec, &workerParam, 7000)
	if err != nil {
		return activePods, activeDeployments, fmt.Errorf("failed to create reid workers deployment: %w", err)
	}
	activeDeployments++

	//STEP 6 - Create edgemesh service for OD
	_, err = CreateEdgeMeshService(mc.kubeClient, service, MultiObjectTrackingWorker, 7000)
	if err != nil {
		return activePods, activeDeployments, fmt.Errorf("failed to create edgemesh service: %w", err)
	}
	activePods++

	return activePods, activeDeployments, err

}

// func (mc *MultiEdgeTrackingServiceController) createWorker(service *sednav1.MultiEdgeTrackingService) error {
// 	// deliver pod for cloudworker
// 	cloudModelName := service.Spec.ReIDWorker.Model.Name
// 	cloudModel, err := mc.client.Models(service.Namespace).Get(context.Background(), cloudModelName, metav1.GetOptions{})
// 	if err != nil {
// 		return fmt.Errorf("failed to get cloud model %s: %w",
// 			cloudModelName, err)
// 	}

// 	var workerParam WorkerParam

// 	secretName := cloudModel.Spec.CredentialName
// 	var modelSecret *v1.Secret
// 	if secretName != "" {
// 		modelSecret, _ = mc.kubeClient.CoreV1().Secrets(service.Namespace).Get(context.TODO(), secretName, metav1.GetOptions{})
// 	}
// 	workerParam.mounts = append(workerParam.mounts, WorkerMount{
// 		URL: &MountURL{
// 			URL:                   cloudModel.Spec.URL,
// 			Secret:                modelSecret,
// 			DownloadByInitializer: true,
// 		},
// 		Name:    "model",
// 		EnvName: "MODEL_URL",
// 	})

// 	workerParam.env = map[string]string{
// 		"NAMESPACE":    service.Namespace,
// 		"SERVICE_NAME": service.Name,
// 		"WORKER_NAME":  "cloudworker-" + utilrand.String(5),

// 		"REID_MODEL_BIND_PORT": strconv.Itoa(int(reIDPort)),
// 	}

// 	workerParam.workerType = ReID

// 	// create cloud pod
// 	_, err = createPodWithTemplate(mc.kubeClient,
// 		service,
// 		&service.Spec.ReIDWorker.Template,
// 		&workerParam)
// 	return err
// }

// func (mc *MultiEdgeTrackingServiceController) createEdgeWorker(service *sednav1.MultiEdgeTrackingService, reIDPort int32) error {
// 	// deliver pod for edgeworker
// 	ctx := context.Background()
// 	edgeWorker := service.Spec.MultiObjectTrackingWorker
// 	var perr error

// 	for _, edgeNode := range edgeWorker {

// 		edgeModelName := edgeNode.Model.Name
// 		edgeModel, err := mc.client.Models(service.Namespace).Get(ctx, edgeModelName, metav1.GetOptions{})
// 		if err != nil {
// 			return fmt.Errorf("failed to get edge model %s: %w",
// 				edgeModelName, err)
// 		}

// 		secretName := edgeModel.Spec.CredentialName
// 		var modelSecret *v1.Secret
// 		if secretName != "" {
// 			modelSecret, _ = mc.kubeClient.CoreV1().Secrets(service.Namespace).Get(context.TODO(), secretName, metav1.GetOptions{})
// 		}

// 		// get ReidIP from nodeName in cloudWorker
// 		//reIDIP, err := GetNodeIPByLabel(mc.kubeClient, service.Spec.ReIDWorker.Template.Spec.NodeSelector, service.Namespace)
// 		reIDIP, err := GetNodeIPByName(mc.kubeClient, service.Spec.ReIDWorker.Template.Spec.NodeName)
// 		//reIDIP, err := GetNodeIPByLabel(mc.kubeClient, service.Spec.ReIDWorker.Template.Spec.NodeSelector, service.Namespace)
// 		if err != nil {
// 			return fmt.Errorf("failed to get node ip: %w", err)
// 		}

// 		var workerParam WorkerParam

// 		workerParam.mounts = append(workerParam.mounts, WorkerMount{
// 			URL: &MountURL{
// 				URL:                   edgeModel.Spec.URL,
// 				Secret:                modelSecret,
// 				DownloadByInitializer: true,
// 			},
// 			Name:    "model",
// 			EnvName: "MODEL_URL",
// 		})

// 		workerParam.env = map[string]string{
// 			"NAMESPACE":    service.Namespace,
// 			"SERVICE_NAME": service.Name,
// 			"WORKER_NAME":  "edgeworker-" + utilrand.String(5),

// 			"REID_MODEL_BIND_IP":   reIDIP,
// 			"REID_MODEL_BIND_PORT": strconv.Itoa(int(reIDPort)),

// 			"LC_SERVER": mc.cfg.LC.Server,
// 		}

// 		workerParam.workerType = MultiObjectTracking
// 		workerParam.hostNetwork = true

// 		// create edge pod
// 		_, perr = createPodWithTemplate(mc.kubeClient,
// 			service,
// 			&edgeNode.Template,
// 			&workerParam)

// 	}

// 	return perr
// }

// GetName returns the name of the joint inference controller
func (mc *MultiEdgeTrackingServiceController) GetName() string {
	return "MultiEdgeTrackingServiceController"
}

// NewJointController creates a new JointInferenceService controller that keeps the relevant pods
// in sync with their corresponding JointInferenceService objects.
func NewMultiEdgeTrackingServiceController(cfg *config.ControllerConfig) (FeatureControllerI, error) {
	var err error
	namespace := cfg.Namespace
	if namespace == "" {
		namespace = metav1.NamespaceAll
	}

	kubeClient, _ := utils.KubeClient()
	kubecfg, _ := utils.KubeConfig()
	crdclient, _ := clientset.NewForConfig(kubecfg)
	kubeInformerFactory := kubeinformers.NewSharedInformerFactoryWithOptions(kubeClient, time.Second*30, kubeinformers.WithNamespace(namespace))

	podInformer := kubeInformerFactory.Core().V1().Pods()

	serviceInformerFactory := informers.NewSharedInformerFactoryWithOptions(crdclient, time.Second*30, informers.WithNamespace(namespace))
	serviceInformer := serviceInformerFactory.Sedna().V1alpha1().MultiEdgeTrackingServices()

	deploymentInformer := kubeInformerFactory.Apps().V1().Deployments()

	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: kubeClient.CoreV1().Events("")})

	mc := &MultiEdgeTrackingServiceController{
		kubeClient: kubeClient,
		client:     crdclient.SednaV1alpha1(),

		queue:    workqueue.NewNamedRateLimitingQueue(workqueue.NewItemExponentialFailureRateLimiter(DefaultBackOff, MaxBackOff), "multiedgetrackingservice"),
		recorder: eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "multiedgetrackingservice-controller"}),
		cfg:      cfg,
	}

	serviceInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			mc.enqueueController(obj, true)
		},

		UpdateFunc: func(old, cur interface{}) {
			mc.enqueueController(cur, true)
		},

		DeleteFunc: func(obj interface{}) {
			mc.enqueueController(obj, true)
		},
	})

	mc.serviceLister = serviceInformer.Lister()
	mc.serviceStoreSynced = serviceInformer.Informer().HasSynced

	podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    mc.addPod,
		UpdateFunc: mc.updatePod,
		DeleteFunc: mc.deletePod,
	})

	mc.podStore = podInformer.Lister()
	mc.podStoreSynced = podInformer.Informer().HasSynced

	deploymentInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    mc.addDeployment,
		UpdateFunc: mc.updateDeployment,
		DeleteFunc: mc.deleteDeployment,
	})

	mc.deploymentsLister = deploymentInformer.Lister()
	mc.deploymentsSynced = deploymentInformer.Informer().HasSynced

	stopCh := messageContext.Done()
	kubeInformerFactory.Start(stopCh)
	serviceInformerFactory.Start(stopCh)
	return mc, err
}
