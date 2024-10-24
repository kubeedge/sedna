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

package objectsearch

import (
	"context"
	"fmt"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilrand "k8s.io/apimachinery/pkg/util/rand"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
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
	sednaclientset "github.com/kubeedge/sedna/pkg/client/clientset/versioned/typed/sedna/v1alpha1"
	sednav1listers "github.com/kubeedge/sedna/pkg/client/listers/sedna/v1alpha1"
	"github.com/kubeedge/sedna/pkg/globalmanager/config"
	"github.com/kubeedge/sedna/pkg/globalmanager/runtime"
)

const (
	// Name is this controller name
	Name = "ObjectSearch"

	// KindName is the kind name of CR this controller controls
	KindName = "ObjectSearchService"
)

const (
	objectSearchUserWorker     = "userworker"
	objectSearchTrackingWorker = "trackingworker"
	objectSearchReidWorker     = "reidworker"
	reidServicePort            = 9378
	userWorkerPort             = 9379
)

// Kind contains the schema.GroupVersionKind for this controller type.
var Kind = sednav1.SchemeGroupVersion.WithKind(KindName)

// Controller ensures that all ObjectSearchService objects
// have corresponding pods to run their configured workload.
type Controller struct {
	kubeClient kubernetes.Interface
	client     sednaclientset.SednaV1alpha1Interface

	// podStoreSynced returns true if the pod store has been synced at least once.
	podStoreSynced cache.InformerSynced
	// A store of pods
	podStore corelisters.PodLister

	// deploymentsSynced returns true if the deployment store has been synced at least once.
	deploymentsSynced cache.InformerSynced
	// A store of deployment
	deploymentsLister appslisters.DeploymentLister

	// serviceStoreSynced returns true if the ObjectSearchService store has been synced at least once.
	serviceStoreSynced cache.InformerSynced
	// A store of service
	serviceLister sednav1listers.ObjectSearchServiceLister

	// ObjectSearchServices that need to be updated
	queue workqueue.RateLimitingInterface

	recorder record.EventRecorder

	cfg *config.ControllerConfig

	sendToEdgeFunc runtime.DownstreamSendFunc
}

// Run starts the main goroutine responsible for watching and syncing services.
func (c *Controller) Run(stopCh <-chan struct{}) {
	workers := 1

	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()

	klog.Infof("Starting %s controller", Name)
	defer klog.Infof("Shutting down %s controller", Name)

	if !cache.WaitForNamedCacheSync(Name, stopCh, c.podStoreSynced, c.serviceStoreSynced) {
		klog.Errorf("failed to wait for %s caches to sync", Name)

		return
	}

	klog.Infof("Starting %s workers", Name)
	for i := 0; i < workers; i++ {
		go wait.Until(c.worker, time.Second, stopCh)
	}

	<-stopCh
}

// enqueueByPod enqueues the ObjectSearchService object of the specified pod.
func (c *Controller) enqueueByPod(pod *v1.Pod, immediate bool) {
	controllerRef := metav1.GetControllerOf(pod)

	if controllerRef == nil {
		return
	}

	if controllerRef.Kind != Kind.Kind {
		return
	}

	service, err := c.serviceLister.ObjectSearchServices(pod.Namespace).Get(controllerRef.Name)
	if err != nil {
		return
	}

	if service.UID != controllerRef.UID {
		return
	}

	c.enqueueController(service, immediate)
}

// When a pod is created, enqueue the controller that manages it and update it's expectations.
func (c *Controller) addPod(obj interface{}) {
	pod := obj.(*v1.Pod)
	if pod.DeletionTimestamp != nil {
		// on a restart of the controller, it's possible a new pod shows up in a state that
		// is already pending deletion. Prevent the pod from being a creation observation.
		c.deletePod(pod)
		return
	}

	// backoff to queue when PodFailed
	immediate := pod.Status.Phase != v1.PodFailed

	c.enqueueByPod(pod, immediate)
}

// When a pod is updated, figure out what object search service manage it and wake them up.
func (c *Controller) updatePod(old, cur interface{}) {
	curPod := cur.(*v1.Pod)
	oldPod := old.(*v1.Pod)

	// no pod update, no queue
	if curPod.ResourceVersion == oldPod.ResourceVersion {
		return
	}

	c.addPod(curPod)
}

// deletePod enqueues the ObjectSearchService obj When a pod is deleted
func (c *Controller) deletePod(obj interface{}) {
	pod, ok := obj.(*v1.Pod)

	// comment from https://github.com/kubernetes/kubernetes/blob/master/pkg/controller/job/job_controller.go

	// When a delete is dropped, the relist will notice a pod in the store not
	// in the list, leading to the insertion of a tombstone object which contains
	// the deleted key/value. Note that this value might be stale. If the pod
	// changed labels the new ObjectSearchService will not be woken up till the periodic resync.
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
	c.enqueueByPod(pod, true)
}

// enqueueByDeployment enqueues the ObjectSearchService object of the specified deployment.
func (c *Controller) enqueueByDeployment(deployment *appsv1.Deployment, immediate bool) {
	controllerRef := metav1.GetControllerOf(deployment)

	if controllerRef == nil {
		return
	}

	if controllerRef.Kind != Kind.Kind {
		return
	}

	service, err := c.serviceLister.ObjectSearchServices(deployment.Namespace).Get(controllerRef.Name)
	if err != nil {
		return
	}

	if service.UID != controllerRef.UID {
		return
	}

	c.enqueueController(service, immediate)
}

// When a deployment is created, enqueue the controller that manages it and update it's expectations.
func (c *Controller) addDeployment(obj interface{}) {
	deployment := obj.(*appsv1.Deployment)
	c.enqueueByDeployment(deployment, true)
}

// deleteDeployment enqueues the ObjectSearchService obj When a deleteDeployment is deleted
func (c *Controller) deleteDeployment(obj interface{}) {
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
	c.enqueueByDeployment(deployment, true)
}

// When a deployment is updated, figure out what object search service manage it and wake them up.
func (c *Controller) updateDeployment(old, cur interface{}) {
	oldD := old.(*appsv1.Deployment)
	curD := cur.(*appsv1.Deployment)
	// no deployment update, no queue
	if curD.ResourceVersion == oldD.ResourceVersion {
		return
	}

	c.addDeployment(curD)
}

// obj could be an *sednav1.ObjectSearchService, or a DeletionFinalStateUnknown marker item,
// immediate tells the controller to update the status right away, and should
// happen ONLY when there was a successful pod run.
func (c *Controller) enqueueController(obj interface{}, immediate bool) {
	key, err := k8scontroller.KeyFunc(obj)
	if err != nil {
		klog.Warningf("Couldn't get key for object %+v: %v", obj, err)
		return
	}

	backoff := time.Duration(0)
	if !immediate {
		backoff = runtime.GetBackoff(c.queue, key)
	}
	c.queue.AddAfter(key, backoff)
}

// worker runs a worker thread that just dequeues items, processes them, and marks them done.
// It enforces that the sync is never invoked concurrently with the same key.
func (c *Controller) worker() {
	for c.processNextWorkItem() {
	}
}

func (c *Controller) processNextWorkItem() bool {
	key, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(key)

	forget, err := c.sync(key.(string))
	if err == nil {
		if forget {
			c.queue.Forget(key)
		}
		return true
	}

	klog.Warningf("Error syncing objectsearch service: %v", err)
	c.queue.AddRateLimited(key)

	return true
}

// sync will sync the objectsearchservice with the given key.
// This function is not meant to be invoked concurrently with the same key.
func (c *Controller) sync(key string) (bool, error) {
	startTime := time.Now()
	defer func() {
		klog.V(4).Infof("Finished syncing objectsearch service %q (%v)", key, time.Since(startTime))
	}()

	ns, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return false, err
	}
	if len(ns) == 0 || len(name) == 0 {
		return false, fmt.Errorf("invalid objectsearch service key %q: either namespace or name is missing", key)
	}
	sharedService, err := c.serviceLister.ObjectSearchServices(ns).Get(name)
	if err != nil {
		if errors.IsNotFound(err) {
			klog.V(4).Infof("ObjectSearchService has been deleted: %v", key)
			return true, nil
		}
		return false, err
	}

	service := *sharedService

	// if service was finished previously, we don't want to redo the termination.
	if isServiceFinished(&service) {
		return true, nil
	}

	// set kind for service in case that the kind is None.
	// more details at https://github.com/kubernetes/kubernetes/issues/3030
	service.SetGroupVersionKind(Kind)

	selectorPods, _ := runtime.GenerateWorkerSelector(&service, objectSearchTrackingWorker)
	selectorDeployments, _ := runtime.GenerateSelector(&service)
	pods, err := c.podStore.Pods(service.Namespace).List(selectorPods)
	deployments, err := c.deploymentsLister.Deployments(service.Namespace).List(selectorDeployments)
	if err != nil {
		return false, err
	}

	latestConditionLen := len(service.Status.Conditions)

	var podFailed int32 = 0
	var deploymentFailed int32 = 0

	// neededPodCounts indicates the num of tracking worker pods should be created successfully in a objectsearch service currently.
	// neededDeploymentCounts indicates the num of deployments should be created successfully in a objectsearch service currently,
	// and one deployment is for userWorker and the other deployment is for reidWorkers.
	var neededPodCounts = int32(len(service.Spec.TrackingWorkers))
	var neededDeploymentCounts int32 = 2

	activePods := runtime.CalcActivePodCount(pods)
	activeDeployments := runtime.CalcActiveDeploymentCount(deployments)

	if service.Status.StartTime == nil {
		now := metav1.Now()
		service.Status.StartTime = &now
	} else {
		podFailed = neededPodCounts - activePods
		deploymentFailed = neededDeploymentCounts - activeDeployments
	}

	var manageServiceErr error
	serviceFailed := false

	var latestConditionType sednav1.ObjectSearchServiceConditionType = ""

	// get the latest condition type
	// based on that condition updated is appended, not inserted.
	jobConditions := service.Status.Conditions
	if len(jobConditions) > 0 {
		latestConditionType = (jobConditions)[len(jobConditions)-1].Type
	}

	var newCondtionType sednav1.ObjectSearchServiceConditionType
	var reason string
	var message string

	switch {
	case podFailed > 0:
		serviceFailed = true
		reason = "podFailed"
		message = "the worker of service failed"
		newCondtionType = sednav1.ObjectSearchServiceCondFailed
		c.recorder.Event(&service, v1.EventTypeWarning, reason, message)
	case deploymentFailed > 0:
		serviceFailed = true
		reason = "deploymentFailed"
		message = "the worker of service failed"
		newCondtionType = sednav1.ObjectSearchServiceCondFailed
		c.recorder.Event(&service, v1.EventTypeWarning, reason, message)
	default:
		if len(pods) == 0 && len(deployments) == 0 {
			activePods, activeDeployments, manageServiceErr = c.createWorkers(&service)
		}
		if manageServiceErr != nil {
			klog.V(2).Infof("failed to create worker: %v", manageServiceErr)
			serviceFailed = true
			message = error.Error(manageServiceErr)
			newCondtionType = sednav1.ObjectSearchServiceCondFailed
			podFailed = neededPodCounts - activePods
			deploymentFailed = neededDeploymentCounts - activeDeployments
		} else {
			// TODO: handle the case that the pod phase is PodSucceeded
			newCondtionType = sednav1.ObjectSearchServiceCondRunning
		}
	}

	if newCondtionType != latestConditionType {
		service.Status.Conditions = append(service.Status.Conditions, newServiceCondition(newCondtionType, reason, message))
	}
	forget := false
	// calculate the number of active pods and deployments
	active := activePods + activeDeployments
	failed := podFailed + deploymentFailed
	// no need to update the objectsearchservice if the status hasn't changed since last time
	if service.Status.Active != active || service.Status.Failed != failed || len(service.Status.Conditions) != latestConditionLen {
		service.Status.Active = active
		service.Status.Failed = failed

		if err := c.updateStatus(&service); err != nil {
			return forget, err
		}

		if serviceFailed && !isServiceFinished(&service) {
			// returning an error will re-enqueue objectsearchservice after the backoff period
			return forget, fmt.Errorf("failed pod(s) detected for objectsearch service key %q", key)
		}

		forget = true
	}

	return forget, manageServiceErr
}

// newServiceCondition creates a new condition
func newServiceCondition(conditionType sednav1.ObjectSearchServiceConditionType, reason, message string) sednav1.ObjectSearchServiceCondition {
	return sednav1.ObjectSearchServiceCondition{
		Type:               conditionType,
		Status:             v1.ConditionTrue,
		LastHeartbeatTime:  metav1.Now(),
		LastTransitionTime: metav1.Now(),
		Reason:             reason,
		Message:            message,
	}
}

// countPods returns number of succeeded and failed pods
func countPods(pods []*v1.Pod) (failed int32) {
	failed = int32(filterPods(pods, v1.PodFailed))
	return
}

// filterPods returns pods based on their phase.
func filterPods(pods []*v1.Pod, phase v1.PodPhase) int {
	result := 0
	for i := range pods {
		if phase == pods[i].Status.Phase {
			result++
		}
	}
	return result
}

func (c *Controller) updateStatus(service *sednav1.ObjectSearchService) error {
	client := c.client.ObjectSearchServices(service.Namespace)
	return runtime.RetryUpdateStatus(service.Name, service.Namespace, func() error {
		newService, err := client.Get(context.TODO(), service.Name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		newService.Status = service.Status
		_, err = client.UpdateStatus(context.TODO(), newService, metav1.UpdateOptions{})
		return err
	})
}

func isServiceFinished(j *sednav1.ObjectSearchService) bool {
	for _, c := range j.Status.Conditions {
		if (c.Type == sednav1.ObjectSearchServiceCondFailed) && c.Status == v1.ConditionTrue {
			return true
		}
	}
	return false
}

func (c *Controller) createWorkers(service *sednav1.ObjectSearchService) (activePods int32, activeDeployments int32, err error) {
	activePods = 0
	activeDeployments = 0

	// create reid worker deployment
	var reidWorkerParam runtime.WorkerParam
	reidWorkerParam.WorkerType = objectSearchReidWorker
	_, err = runtime.CreateDeploymentWithTemplate(c.kubeClient, service, &service.Spec.ReidWorkers.DeploymentSpec, &reidWorkerParam)
	if err != nil {
		return activePods, activeDeployments, fmt.Errorf("failed to create reid worker deployment: %w", err)
	}
	activeDeployments++

	// create reid worker edgemesh service
	reidServiceHost, err := runtime.CreateEdgeMeshService(c.kubeClient, service, objectSearchReidWorker, reidServicePort)
	if err != nil {
		return activePods, activeDeployments, fmt.Errorf("failed to create reid worker edgemesh service: %w", err)
	}

	reidServiceURL := fmt.Sprintf("%s:%d", reidServiceHost, reidServicePort)

	// create user worker deployment
	userWorkerReplicas := int32(1)
	userWorkerDeployment := &appsv1.DeploymentSpec{
		Replicas: &userWorkerReplicas,
		Template: service.Spec.UserWorker.Template,
	}
	var userWorkerParam runtime.WorkerParam
	userWorkerParam.WorkerType = objectSearchUserWorker
	userWorkerParam.Env = map[string]string{
		"NAMESPACE":    service.Namespace,
		"SERVICE_NAME": service.Name,
		"WORKER_NAME":  "userworker-" + utilrand.String(5),
	}
	_, err = runtime.CreateDeploymentWithTemplate(c.kubeClient, service, userWorkerDeployment, &userWorkerParam)

	if err != nil {
		return activePods, activeDeployments, fmt.Errorf("failed to create user worker: %w", err)
	}
	activeDeployments++

	// create user worker service
	userWorkerHost, err := runtime.CreateEdgeMeshService(c.kubeClient, service, objectSearchUserWorker, userWorkerPort)
	if err != nil {
		return activePods, activeDeployments, fmt.Errorf("failed to create edgemesh service: %w", err)
	}
	userWorkerURL := fmt.Sprintf("%s:%d", userWorkerHost, userWorkerPort)

	// create tracking worker pods
	var trackingWorkerParam runtime.WorkerParam
	trackingWorkerParam.WorkerType = objectSearchTrackingWorker
	for i, trackingWorker := range service.Spec.TrackingWorkers {
		trackingWorkerParam.Env = map[string]string{
			"NAMESPACE":      service.Namespace,
			"SERVICE_NAME":   service.Name,
			"WORKER_NAME":    "trackingworker-" + utilrand.String(5),
			"USERWORKER_URL": userWorkerURL,
			"EDGEMESH_URL":   reidServiceURL,
		}
		_, err = runtime.CreatePodWithTemplate(c.kubeClient, service, &trackingWorker.Template, &trackingWorkerParam)
		if err != nil {
			return activePods, activeDeployments, fmt.Errorf("failed to create %dth tracking worker: %w", i, err)
		}
		activePods++
	}

	return activePods, activeDeployments, err
}

// New creates a new ObjectSearchService controller that keeps the relevant pods
// in sync with their corresponding ObjectSearchService objects.
func New(cc *runtime.ControllerContext) (runtime.FeatureControllerI, error) {
	cfg := cc.Config

	podInformer := cc.KubeInformerFactory.Core().V1().Pods()

	deploymentInformer := cc.KubeInformerFactory.Apps().V1().Deployments()

	serviceInformer := cc.SednaInformerFactory.Sedna().V1alpha1().ObjectSearchServices()

	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: cc.KubeClient.CoreV1().Events("")})

	jc := &Controller{
		kubeClient: cc.KubeClient,
		client:     cc.SednaClient.SednaV1alpha1(),

		queue:    workqueue.NewNamedRateLimitingQueue(workqueue.NewItemExponentialFailureRateLimiter(runtime.DefaultBackOff, runtime.MaxBackOff), "objectsearchservice"),
		recorder: eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "objectsearch-controller"}),
		cfg:      cfg,
	}

	serviceInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			jc.enqueueController(obj, true)
		},

		UpdateFunc: func(old, cur interface{}) {
			jc.enqueueController(cur, true)
		},

		DeleteFunc: func(obj interface{}) {
			jc.enqueueController(obj, true)
		},
	})

	jc.serviceLister = serviceInformer.Lister()
	jc.serviceStoreSynced = serviceInformer.Informer().HasSynced

	podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    jc.addPod,
		UpdateFunc: jc.updatePod,
		DeleteFunc: jc.deletePod,
	})

	jc.podStore = podInformer.Lister()
	jc.podStoreSynced = podInformer.Informer().HasSynced

	deploymentInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    jc.addDeployment,
		UpdateFunc: jc.updateDeployment,
		DeleteFunc: jc.deleteDeployment,
	})
	jc.deploymentsLister = deploymentInformer.Lister()
	jc.deploymentsSynced = deploymentInformer.Informer().HasSynced

	return jc, nil
}
