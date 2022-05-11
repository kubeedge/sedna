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

package featureextraction

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
	FEWorker = "fe"
	FEPort   = 6000
)

const (
	// Name is this controller name
	Name = "FeatureExtraction"

	// KindName is the kind name of CR this controller controls
	KindName = "FeatureExtractionService"
)

// FeatureExtractionServicerKind contains the schema.GroupVersionKind for this controller type.
var FeatureExtractionServiceKind = sednav1.SchemeGroupVersion.WithKind("FeatureExtractionService")

// Controller ensures that all FeatureExtractionService objects
// have corresponding pods to run their configured workload.
type Controller struct {
	kubeClient kubernetes.Interface
	client     sednaclientset.SednaV1alpha1Interface

	// podStoreSynced returns true if the pod store has been synced at least once.
	podStoreSynced cache.InformerSynced
	// A store of pods
	podStore corelisters.PodLister

	// serviceStoreSynced returns true if the FeatureExtractionService store has been synced at least once.
	serviceStoreSynced cache.InformerSynced
	// A store of service
	serviceLister sednav1listers.FeatureExtractionServiceLister

	// deploymentsSynced returns true if the deployment store has been synced at least once.
	deploymentsSynced cache.InformerSynced
	// A store of deployment
	deploymentsLister appslisters.DeploymentLister

	enqueueDeployment func(deployment *appsv1.Deployment)

	// FeatureExtractionService that need to be updated
	queue workqueue.RateLimitingInterface

	recorder record.EventRecorder

	cfg *config.ControllerConfig

	sendToEdgeFunc func(nodeName string, eventType watch.EventType, obj interface{}) error
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

func (c *Controller) enqueueByDeployment(deployment *appsv1.Deployment) {
	controllerRef := metav1.GetControllerOf(deployment)

	klog.Infof("Deployment enqueued %v", deployment.Kind)

	if controllerRef == nil {
		return
	}

	if controllerRef.Kind != FeatureExtractionServiceKind.Kind {
		return
	}

	service, err := c.serviceLister.FeatureExtractionServices(deployment.Namespace).Get(controllerRef.Name)
	if err != nil {
		return
	}

	if service.UID != controllerRef.UID {
		return
	}

	c.enqueueController(service, true)
}

func (c *Controller) addDeployment(obj interface{}) {
	deployment := obj.(*appsv1.Deployment)
	c.enqueueByDeployment(deployment)
}

//deleteDeployment enqueues the FeatureExtractionService obj When a deleteDeployment is deleted
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
	c.enqueueByDeployment(deployment)
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

// enqueueByPod enqueues the FeatureExtractionService object of the specified pod.
func (c *Controller) enqueueByPod(pod *v1.Pod, immediate bool) {
	controllerRef := metav1.GetControllerOf(pod)

	if controllerRef == nil {
		return
	}

	if controllerRef.Kind != FeatureExtractionServiceKind.Kind {
		return
	}

	service, err := c.serviceLister.FeatureExtractionServices(pod.Namespace).Get(controllerRef.Name)
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

// When a pod is updated, figure out what joint inference service manage it and wake them up.
func (c *Controller) updatePod(old, cur interface{}) {
	curPod := cur.(*v1.Pod)
	oldPod := old.(*v1.Pod)

	// no pod update, no queue
	if curPod.ResourceVersion == oldPod.ResourceVersion {
		return
	}

	c.addPod(curPod)
}

// deletePod enqueues the FeatureExtractionService obj When a pod is deleted
func (c *Controller) deletePod(obj interface{}) {
	pod, ok := obj.(*v1.Pod)

	// comment from https://github.com/kubernetes/kubernetes/blob/master/pkg/controller/job/job_controller.go

	// When a delete is dropped, the relist will notice a pod in the store not
	// in the list, leading to the insertion of a tombstone object which contains
	// the deleted key/value. Note that this value might be stale. If the pod
	// changed labels the new FeatureExtractionService will not be woken up till the periodic resync.
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

	klog.Warningf("Error syncing feature extraction service: %v", err)
	c.queue.AddRateLimited(key)

	return true
}

// sync will sync the FeatureExtractionService with the given key.
// This function is not meant to be invoked concurrently with the same key.
func (c *Controller) sync(key string) (bool, error) {
	startTime := time.Now()
	defer func() {
		klog.V(4).Infof("Finished syncing feature extraction service %q (%v)", key, time.Since(startTime))
	}()

	ns, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return false, err
	}
	if len(ns) == 0 || len(name) == 0 {
		return false, fmt.Errorf("Invalid feature extraction service key %q: either namespace or name is missing", key)
	}
	sharedFeatureExtractionService, err := c.serviceLister.FeatureExtractionServices(ns).Get(name)
	if err != nil {
		if errors.IsNotFound(err) {
			klog.V(4).Infof("FeatureExtractionService has been deleted: %v", key)
			return true, nil
		}
		return false, err
	}

	FeatureExtractionService := *sharedFeatureExtractionService

	if isFeatureExtractionServiceFinished(&FeatureExtractionService) {
		return true, nil
	}

	FeatureExtractionService.SetGroupVersionKind(FeatureExtractionServiceKind)

	selector, _ := runtime.GenerateSelector(&FeatureExtractionService)
	pods, err := c.podStore.Pods(FeatureExtractionService.Namespace).List(selector)
	deployment, err := c.deploymentsLister.Deployments(FeatureExtractionService.Namespace).List(selector)

	if err != nil {
		return false, err
	}

	klog.V(4).Infof("list MOT service %v/%v, %v pods: %v", FeatureExtractionService.Namespace, FeatureExtractionService.Name, len(pods), pods)

	latestConditionLen := len(FeatureExtractionService.Status.Conditions)

	activePods := runtime.CalcActivePodCount(pods)
	activeDeployments := runtime.CalcActiveDeploymentCount(deployment)

	var failedPods, failedDeployment int32 = 0, 0

	var neededPodCounts int32 = *sharedFeatureExtractionService.Spec.Replicas

	var neededDeploymentCounts int32 = int32(reflect.TypeOf(sednav1.FeatureExtractionServiceSpec{}).NumField())

	// first start
	if FeatureExtractionService.Status.StartTime == nil {
		now := metav1.Now()
		FeatureExtractionService.Status.StartTime = &now
	} else {
		failedPods = neededPodCounts - activePods
		failedDeployment = neededDeploymentCounts - activeDeployments
	}

	var manageServiceErr error
	serviceFailed := false

	var latestConditionType sednav1.FeatureExtractionServiceConditionType = ""

	// get the latest condition type
	// based on that condition updated is appended, not inserted.
	jobConditions := FeatureExtractionService.Status.Conditions
	if len(jobConditions) > 0 {
		latestConditionType = (jobConditions)[len(jobConditions)-1].Type
	}

	var newCondtionType sednav1.FeatureExtractionServiceConditionType
	var reason string
	var message string

	if failedPods > 0 || failedDeployment > 0 {
		serviceFailed = true
		// TODO: Split code to handle deployment failure separately
		// TODO: get the failed worker, and knows that which worker fails, edge inference worker or cloud inference worker
		reason = "workerFailed"
		message = "the worker of FeatureExtractionService failed"
		newCondtionType = sednav1.FeatureExtractionServiceCondFailed
		c.recorder.Event(&FeatureExtractionService, v1.EventTypeWarning, reason, message)
	} else {
		if len(pods) == 0 {
			activePods, activeDeployments, manageServiceErr = c.createWorkers(&FeatureExtractionService)
		}
		if manageServiceErr != nil {
			serviceFailed = true
			message = error.Error(manageServiceErr)
			newCondtionType = sednav1.FeatureExtractionServiceCondFailed
			failedPods = neededPodCounts - activePods
			failedDeployment = neededDeploymentCounts - activeDeployments
		} else {
			// TODO: handle the case that the pod phase is PodSucceeded
			newCondtionType = sednav1.FeatureExtractionServiceCondRunning
		}
	}

	//
	if newCondtionType != latestConditionType {
		FeatureExtractionService.Status.Conditions = append(FeatureExtractionService.Status.Conditions, NewFeatureExtractionServiceCondition(newCondtionType, reason, message))
	}
	forget := false

	// no need to update the FeatureExtractionService if the status hasn't changed since last time
	if FeatureExtractionService.Status.Active != activePods || FeatureExtractionService.Status.Failed != failedPods || len(FeatureExtractionService.Status.Conditions) != latestConditionLen {
		FeatureExtractionService.Status.Active = activePods
		FeatureExtractionService.Status.Failed = failedPods

		if err := c.updateStatus(&FeatureExtractionService); err != nil {
			return forget, err
		}

		if serviceFailed && !isFeatureExtractionServiceFinished(&FeatureExtractionService) {
			// returning an error will re-enqueue FeatureExtractionService after the backoff period
			return forget, fmt.Errorf("failed pod(s) detected for MOT service key %q", key)
		}

		forget = true
	}

	return forget, manageServiceErr
}

// NewFeatureExtractionServiceCondition creates a new joint condition
func NewFeatureExtractionServiceCondition(conditionType sednav1.FeatureExtractionServiceConditionType, reason, message string) sednav1.FeatureExtractionServiceCondition {
	return sednav1.FeatureExtractionServiceCondition{
		Type:               conditionType,
		Status:             v1.ConditionTrue,
		LastHeartbeatTime:  metav1.Now(),
		LastTransitionTime: metav1.Now(),
		Reason:             reason,
		Message:            message,
	}
}

func (c *Controller) updateStatus(service *sednav1.FeatureExtractionService) error {
	client := c.client.FeatureExtractionServices(service.Namespace)
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

func isFeatureExtractionServiceFinished(j *sednav1.FeatureExtractionService) bool {
	for _, c := range j.Status.Conditions {
		if (c.Type == sednav1.FeatureExtractionServiceCondFailed) && c.Status == v1.ConditionTrue {
			return true
		}
	}
	return false
}

func (c *Controller) createFeatureExtractionWorker(service *sednav1.FeatureExtractionService, activePods *int32, activeDeployments *int32) (err error) {
	var workerParam runtime.WorkerParam
	var secretName string
	var modelSecret *v1.Secret

	/*

		FE DEPLOYMENT

	*/

	// Create parameters that will be used in the deployment
	workerParam.WorkerType = FEWorker

	// Load model used by the pods in this deployment
	feModelName := service.Spec.Model.Name
	feModel, err := c.client.Models(service.Namespace).Get(context.Background(), feModelName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("Failed to get model %s: %w", feModelName, err)
	}

	secretName = feModel.Spec.CredentialName
	if secretName != "" {
		modelSecret, _ = c.kubeClient.CoreV1().Secrets(service.Namespace).Get(context.Background(), secretName, metav1.GetOptions{})
	}

	workerParam.Mounts = append(make([]runtime.WorkerMount, 1), runtime.WorkerMount{
		URL: &runtime.MountURL{
			URL:                   feModel.Spec.URL,
			Secret:                modelSecret,
			DownloadByInitializer: true,
		},
		Name:    "model",
		EnvName: "MODEL_URL",
	})

	workerParam.Env = map[string]string{
		"WORKER_NAME":  utilrand.String(5),
		"NAMESPACE":    service.Namespace,
		"SERVICE_NAME": service.Name,
		"LC_SERVER":    c.cfg.LC.Server,
	}

	if service.Spec.KafkaSupport {
		workerParam.Env["KAFKA_ENABLED"] = strconv.FormatBool(service.Spec.KafkaSupport)
	}

	// Create FE deployment AND related pods (as part of the deployment creation)
	_, err = runtime.CreateDeploymentWithTemplate(c.kubeClient, service, &service.Spec.DeploymentSpec, &workerParam, FEPort)
	if err != nil {
		return fmt.Errorf("failed to create feature extraction deployment: %w", err)
	}

	// Create edgemesh service for FE
	_, err = runtime.CreateEdgeMeshService(c.kubeClient, service, FEWorker, FEPort)
	if err != nil {
		return fmt.Errorf("failed to create edgemesh service: %w", err)
	}

	*activeDeployments++

	*activePods++

	return nil
}

func (c *Controller) createWorkers(service *sednav1.FeatureExtractionService) (activePods int32, activeDeployments int32, err error) {
	activePods = 0
	activeDeployments = 0

	err = c.createFeatureExtractionWorker(service, &activePods, &activeDeployments)
	if err != nil {
		return activePods, activeDeployments, fmt.Errorf("failed to create feature extraction service: %w", err)
	}

	return activePods, activeDeployments, nil

}

func (c *Controller) GetName() string {
	return "FeatureExtractionController"
}

// NewJointController creates a new FeatureExtractionService controller that keeps the relevant pods
// in sync with their corresponding FeatureExtractionService objects.
func New(cc *runtime.ControllerContext) (runtime.FeatureControllerI, error) {
	cfg := cc.Config

	podInformer := cc.KubeInformerFactory.Core().V1().Pods()

	serviceInformer := cc.SednaInformerFactory.Sedna().V1alpha1().FeatureExtractionServices()

	deploymentInformer := cc.KubeInformerFactory.Apps().V1().Deployments()

	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: cc.KubeClient.CoreV1().Events("")})

	c := &Controller{
		kubeClient: cc.KubeClient,
		client:     cc.SednaClient.SednaV1alpha1(),

		queue:    workqueue.NewNamedRateLimitingQueue(workqueue.NewItemExponentialFailureRateLimiter(runtime.DefaultBackOff, runtime.MaxBackOff), "featureextractionservice"),
		recorder: eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "featureextractionservice-controller"}),
		cfg:      cfg,
	}

	serviceInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			c.enqueueController(obj, true)
			c.syncToEdge(watch.Added, obj)
		},

		UpdateFunc: func(old, cur interface{}) {
			c.enqueueController(cur, true)
			c.syncToEdge(watch.Added, cur)
		},

		DeleteFunc: func(obj interface{}) {
			c.enqueueController(obj, true)
			c.syncToEdge(watch.Deleted, obj)
		},
	})

	c.serviceLister = serviceInformer.Lister()
	c.serviceStoreSynced = serviceInformer.Informer().HasSynced

	podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    c.addPod,
		UpdateFunc: c.updatePod,
		DeleteFunc: c.deletePod,
	})

	c.podStore = podInformer.Lister()
	c.podStoreSynced = podInformer.Informer().HasSynced

	deploymentInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    c.addDeployment,
		UpdateFunc: c.updateDeployment,
		DeleteFunc: c.deleteDeployment,
	})

	c.deploymentsLister = deploymentInformer.Lister()
	c.deploymentsSynced = deploymentInformer.Informer().HasSynced

	return c, nil
}
