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

package jointinference

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strconv"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
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
	// Name is this controller name
	Name = "JointInference"

	// KindName is the kind name of CR this controller controls
	KindName = "JointInferenceService"
)

const (
	jointInferenceForEdge  = "Edge"
	jointInferenceForCloud = "Cloud"
	BigModelPort           = 5000
)

// Kind contains the schema.GroupVersionKind for this controller type.
var Kind = sednav1.SchemeGroupVersion.WithKind(KindName)

// Controller ensures that all JointInferenceService objects
// have corresponding pods to run their configured workload.
type Controller struct {
	kubeClient kubernetes.Interface
	client     sednaclientset.SednaV1alpha1Interface

	// podStoreSynced returns true if the pod store has been synced at least once.
	podStoreSynced cache.InformerSynced
	// A store of pods
	podStore corelisters.PodLister

	// serviceStoreSynced returns true if the JointInferenceService store has been synced at least once.
	serviceStoreSynced cache.InformerSynced
	// A store of service
	serviceLister sednav1listers.JointInferenceServiceLister

	//deploymentStoreSynced returns true if the deployment store has been synced at least once.
	deploymentStoreSynced cache.InformerSynced
	// A store of deployment
	deploymentsLister appslisters.DeploymentLister

	// JointInferenceServices that need to be updated
	queue workqueue.RateLimitingInterface

	recorder record.EventRecorder

	cfg *config.ControllerConfig

	sendToEdgeFunc runtime.DownstreamSendFunc

	bigModelHost string

	selector labels.Selector
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

// enqueueByPod enqueues the JointInferenceService object of the specified pod.
func (c *Controller) enqueueByPod(pod *v1.Pod, immediate bool) {
	controllerRef := metav1.GetControllerOf(pod)

	if controllerRef == nil {
		return
	}

	if controllerRef.Kind != Kind.Kind {
		return
	}

	service, err := c.serviceLister.JointInferenceServices(pod.Namespace).Get(controllerRef.Name)
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

// deletePod enqueues the JointinferenceService obj When a pod is deleted
func (c *Controller) deletePod(obj interface{}) {
	pod, ok := obj.(*v1.Pod)

	// comment from https://github.com/kubernetes/kubernetes/blob/master/pkg/controller/job/job_controller.go

	// When a delete is dropped, the relist will notice a pod in the store not
	// in the list, leading to the insertion of a tombstone object which contains
	// the deleted key/value. Note that this value might be stale. If the pod
	// changed labels the new JointInferenceService will not be woken up till the periodic resync.
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

// obj could be an *sednav1.JointInferenceService, or a DeletionFinalStateUnknown marker item,
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

	klog.Warningf("Error syncing jointinference service: %v", err)
	c.queue.AddRateLimited(key)

	return true
}

// sync will sync the jointinferenceservice with the given key.
// This function is not meant to be invoked concurrently with the same key.
func (c *Controller) sync(key string) (bool, error) {
	startTime := time.Now()
	defer func() {
		klog.V(4).Infof("Finished syncing jointinference service %q (%v)", key, time.Since(startTime))
	}()

	ns, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return false, err
	}
	if len(ns) == 0 || len(name) == 0 {
		return false, fmt.Errorf("invalid jointinference service key %q: either namespace or name is missing", key)
	}
	sharedService, err := c.serviceLister.JointInferenceServices(ns).Get(name)
	if err != nil {
		if errors.IsNotFound(err) {
			klog.V(4).Infof("JointInferenceService has been deleted: %v", key)
			return true, nil
		}
		return false, err
	}

	service := *sharedService

	// if service was finished previously, we don't want to redo the termination
	if isServiceFinished(&service) {
		return true, nil
	}

	// set kind for service in case that the kind is None
	// more details at https://github.com/kubernetes/kubernetes/issues/3030
	service.SetGroupVersionKind(Kind)

	c.selector, _ = runtime.GenerateSelector(&service)
	pods, err := c.podStore.Pods(service.Namespace).List(c.selector)
	if err != nil {
		return false, err
	}
	deployments, err := c.deploymentsLister.Deployments(service.Namespace).List(c.selector)

	if err != nil {
		return false, err
	}

	klog.V(4).Infof("list jointinference service %v/%v, %v pods: %v", service.Namespace, service.Name, len(pods), pods)

	latestConditionLen := len(service.Status.Conditions)

	activePods := runtime.CalcActivePodCount(pods)
	activeDeployments := runtime.CalcActiveDeploymentCount(deployments)
	activeCloudPod := false
	activeCloudDeployment := false
	activeEdgePod := false
	activeEdgeDeployment := false

	var failedPods, failedDeployments int32 = 0, 0

	// neededCounts means that two pods should be created successfully in a jointinference service currently
	// two deployments consist of edge pod and cloud pod
	var neededPodCounts int32 = 2
	var neededDeploymentCounts int32 = 2

	if service.Status.StartTime == nil {
		now := metav1.Now()
		service.Status.StartTime = &now
	} else {
		failedPods = neededPodCounts - activePods
		failedDeployments = neededDeploymentCounts - activeDeployments
	}

	var manageServiceErr error
	serviceFailed := false

	var latestConditionType sednav1.JointInferenceServiceConditionType = ""

	// get the latest condition type
	// based on that condition updated is appended, not inserted.
	jobConditions := service.Status.Conditions
	if len(jobConditions) > 0 {
		latestConditionType = (jobConditions)[len(jobConditions)-1].Type
	}

	var newCondtionType sednav1.JointInferenceServiceConditionType
	var reason string
	var message string

	if failedPods > 0 || failedDeployments > 0 {
		serviceFailed = true
		if (!activeCloudPod) || (!activeCloudDeployment) {
			reason = "cloudWorkerFailed\n"
			message = "the cloud worker of service failed\n"
		}
		if (!activeEdgePod) || (!activeEdgeDeployment) {
			reason += "edgeWorkerFailed\n"
			message += "the edge worker of service failed\n"
		}
		newCondtionType = sednav1.JointInferenceServiceCondFailed
		c.recorder.Event(&service, v1.EventTypeWarning, reason, message)
	} else {
		if len(pods) == 0 {
			activePods, activeDeployments, manageServiceErr = c.createWorkers(&service, &activeCloudPod, &activeCloudDeployment, &activeEdgePod, &activeEdgeDeployment)
		}
		if manageServiceErr != nil {
			serviceFailed = true
			message = error.Error(manageServiceErr)
			newCondtionType = sednav1.JointInferenceServiceCondFailed
			failedPods = neededPodCounts - activePods
			failedDeployments = neededDeploymentCounts - activeDeployments
		} else {
			// TODO: handle the case that the pod phase is PodSucceeded
			newCondtionType = sednav1.JointInferenceServiceCondRunning
		}
	}

	//
	if newCondtionType != latestConditionType {
		service.Status.Conditions = append(service.Status.Conditions, newServiceCondition(newCondtionType, reason, message))
	}
	forget := false

	// calculate the number of active pods and deployments
	active := activePods + activeDeployments
	failed := failedPods + failedDeployments
	// no need to update the jointinferenceservice if the status hasn't changed since last time
	if service.Status.Active != active || service.Status.Failed != failed || len(service.Status.Conditions) != latestConditionLen {
		service.Status.Active = activePods
		service.Status.Failed = failedPods

		if err := c.updateStatus(&service); err != nil {
			return forget, err
		}

		if serviceFailed && !isServiceFinished(&service) {
			// returning an error will re-enqueue jointinferenceservice after the backoff period
			return forget, fmt.Errorf("failed pod(s) detected for jointinference service key %q", key)
		}

		forget = true
	}

	return forget, manageServiceErr
}

// newServiceCondition creates a new joint condition
func newServiceCondition(conditionType sednav1.JointInferenceServiceConditionType, reason, message string) sednav1.JointInferenceServiceCondition {
	return sednav1.JointInferenceServiceCondition{
		Type:               conditionType,
		Status:             v1.ConditionTrue,
		LastHeartbeatTime:  metav1.Now(),
		LastTransitionTime: metav1.Now(),
		Reason:             reason,
		Message:            message,
	}
}

func (c *Controller) updateStatus(service *sednav1.JointInferenceService) error {
	client := c.client.JointInferenceServices(service.Namespace)
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

func isServiceFinished(j *sednav1.JointInferenceService) bool {
	for _, c := range j.Status.Conditions {
		if (c.Type == sednav1.JointInferenceServiceCondFailed) && c.Status == v1.ConditionTrue {
			return true
		}
	}
	return false
}

func (c *Controller) createWorkers(service *sednav1.JointInferenceService, activeCloudPod *bool, activeCloudDeployment *bool, activeEdgePod *bool, activeEdgeDeployment *bool) (activePods, activeDeployments int32, err error) {
	var bigModelPort int32 = BigModelPort
	// create cloud worker
	err = c.createCloudWorker(service, bigModelPort)
	if err != nil {
		return activePods, activeDeployments, fmt.Errorf("failed to create cloudWorker: %w", err)
	}
	*activeCloudPod = true
	*activeCloudDeployment = true
	activePods++
	activeDeployments++

	// create k8s service for cloudPod
	c.bigModelHost, err = runtime.CreateEdgeMeshService(c.kubeClient, service, jointInferenceForCloud, bigModelPort)
	if err != nil {
		return activePods, activeDeployments, fmt.Errorf("failed to create edgemesh service: %w", err)
	}

	// create edge worker
	err = c.createEdgeWorker(service, c.bigModelHost, bigModelPort)
	if err != nil {
		return activePods, activeDeployments, fmt.Errorf("failed to create edgeWorker: %w", err)
	}
	*activeEdgePod = true
	*activeEdgeDeployment = true
	activePods++
	activeDeployments++

	return activePods, activeDeployments, err
}

func (c *Controller) createCloudWorker(service *sednav1.JointInferenceService, bigModelPort int32) error {
	// deliver deployment for cloudworker
	cloudModelName := service.Spec.CloudWorker.Model.Name
	cloudModel, err := c.client.Models(service.Namespace).Get(context.Background(), cloudModelName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get cloud model %s: %w",
			cloudModelName, err)
	}

	var workerParam runtime.WorkerParam

	secretName := cloudModel.Spec.CredentialName
	var modelSecret *v1.Secret
	if secretName != "" {
		modelSecret, _ = c.kubeClient.CoreV1().Secrets(service.Namespace).Get(context.TODO(), secretName, metav1.GetOptions{})
	}
	workerParam.Mounts = append(workerParam.Mounts, runtime.WorkerMount{
		URL: &runtime.MountURL{
			URL:                   cloudModel.Spec.URL,
			Secret:                modelSecret,
			DownloadByInitializer: true,
		},
		Name:    "model",
		EnvName: "MODEL_URL",
	})

	workerParam.Env = map[string]string{
		"NAMESPACE":           service.Namespace,
		"SERVICE_NAME":        service.Name,
		"WORKER_NAME":         "cloudworker",
		"BIG_MODEL_BIND_PORT": strconv.Itoa(int(bigModelPort)),
	}

	workerParam.WorkerType = jointInferenceForCloud

	cloudWorkerDeployment := &appsv1.DeploymentSpec{
		Template: service.Spec.CloudWorker.Template,
	}
	// Create cloudWorker deployment AND related pods (as part of the deployment creation)
	_, err = runtime.CreateDeploymentWithTemplate(c.kubeClient,
		service,
		cloudWorkerDeployment,
		&workerParam,
		bigModelPort,
	)
	if err != nil {
		return fmt.Errorf("failed to create cloudWorker deployment: %w", err)
	}
	return nil
}

func (c *Controller) createEdgeWorker(service *sednav1.JointInferenceService, bigModelHost string, bigModelPort int32) error {
	// deliver edge deployment for edgeworker
	ctx := context.Background()
	edgeModelName := service.Spec.EdgeWorker.Model.Name
	edgeModel, err := c.client.Models(service.Namespace).Get(ctx, edgeModelName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get edge model %s: %w",
			edgeModelName, err)
	}

	secretName := edgeModel.Spec.CredentialName
	var modelSecret *v1.Secret
	if secretName != "" {
		modelSecret, _ = c.kubeClient.CoreV1().Secrets(service.Namespace).Get(context.TODO(), secretName, metav1.GetOptions{})
	}

	edgeWorker := service.Spec.EdgeWorker
	HEMParameterJSON, _ := json.Marshal(edgeWorker.HardExampleMining.Parameters)
	HEMParameterString := string(HEMParameterJSON)

	var workerParam runtime.WorkerParam

	workerParam.Mounts = append(workerParam.Mounts, runtime.WorkerMount{
		URL: &runtime.MountURL{
			URL:                   edgeModel.Spec.URL,
			Secret:                modelSecret,
			DownloadByInitializer: true,
		},
		Name:    "model",
		EnvName: "MODEL_URL",
	})

	workerParam.Env = map[string]string{
		"NAMESPACE":    service.Namespace,
		"SERVICE_NAME": service.Name,
		"WORKER_NAME":  "edgeworker",

		"BIG_MODEL_IP":   bigModelHost,
		"BIG_MODEL_PORT": strconv.Itoa(int(bigModelPort)),

		"HEM_NAME":       edgeWorker.HardExampleMining.Name,
		"HEM_PARAMETERS": HEMParameterString,

		"LC_SERVER": c.cfg.LC.Server,
	}

	workerParam.WorkerType = jointInferenceForEdge
	workerParam.HostNetwork = true

	edgeWorkerDeployment := &appsv1.DeploymentSpec{
		Template: service.Spec.EdgeWorker.Template,
	}

	// create edge pod
	_, err = runtime.CreateDeploymentWithTemplate(c.kubeClient,
		service,
		edgeWorkerDeployment,
		&workerParam,
		bigModelPort,
	)
	if err != nil {
		return fmt.Errorf("failed to create edgeWorker deployment: %w", err)
	}

	return nil
}

// New creates a new JointInferenceService controller that keeps the relevant pods
// in sync with their corresponding JointInferenceService objects.
func New(cc *runtime.ControllerContext) (runtime.FeatureControllerI, error) {
	cfg := cc.Config

	podInformer := cc.KubeInformerFactory.Core().V1().Pods()

	serviceInformer := cc.SednaInformerFactory.Sedna().V1alpha1().JointInferenceServices()

	deploymentInformer := cc.KubeInformerFactory.Apps().V1().Deployments()

	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: cc.KubeClient.CoreV1().Events("")})

	jc := &Controller{
		kubeClient: cc.KubeClient,
		client:     cc.SednaClient.SednaV1alpha1(),

		queue:    workqueue.NewNamedRateLimitingQueue(workqueue.NewItemExponentialFailureRateLimiter(runtime.DefaultBackOff, runtime.MaxBackOff), "jointinferenceservice"),
		recorder: eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "jointinferenceservice-controller"}),
		cfg:      cfg,
	}

	serviceInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			jc.enqueueController(obj, true)
			jc.syncToEdge(watch.Added, obj)
		},

		UpdateFunc: jc.updateService,

		DeleteFunc: func(obj interface{}) {
			jc.enqueueController(obj, true)
			jc.syncToEdge(watch.Deleted, obj)
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
	jc.deploymentStoreSynced = deploymentInformer.Informer().HasSynced

	return jc, nil
}

func (c *Controller) addDeployment(obj interface{}) {
	deployment := obj.(*appsv1.Deployment)
	c.enqueueByDeployment(deployment)
}

// deleteDeployment enqueues the JointInferenceService obj When a deleteDeployment is deleted
func (c *Controller) deleteDeployment(obj interface{}) {
	deployment, ok := obj.(*appsv1.Deployment)

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

func (c *Controller) updateDeployment(old, cur interface{}) {
	oldDeployment := old.(*appsv1.Deployment)
	curDeployment := cur.(*appsv1.Deployment)
	// no deployment update, no queue
	if curDeployment.ResourceVersion == oldDeployment.ResourceVersion {
		return
	}
	c.addDeployment(curDeployment)
}

func (c *Controller) enqueueByDeployment(deployment *appsv1.Deployment) {
	controllerRef := metav1.GetControllerOf(deployment)

	klog.Infof("Deployment enqueued %v", deployment.Kind)

	if controllerRef == nil {
		return
	}

	if controllerRef.Kind != Kind.Kind {
		return
	}

	service, err := c.serviceLister.JointInferenceServices(deployment.Namespace).Get(deployment.Name)
	if err != nil {
		return
	}

	if service.UID != controllerRef.UID {
		return
	}

	c.enqueueController(service, true)
}

func (c *Controller) updateService(old, cur interface{}) {
	oldService, ok := old.(*sednav1.JointInferenceService)
	if !ok {
		return
	}
	curService, ok := cur.(*sednav1.JointInferenceService)
	if !ok {
		return
	}

	if oldService == curService {
		return
	}

	if reflect.DeepEqual(oldService.Spec, curService.Spec) {
		return
	}
	// if CRD is changed,and the service.Generation is changed, update deployment settings
	klog.Infof("Service is updated, delete previous deployments")
	curService.SetGroupVersionKind(Kind)
	// if the service.Generation is changed, update deployment settings
	if oldService.Generation != curService.Generation {
		// delete previous deployments
		deployments, err := c.deploymentsLister.Deployments(curService.Namespace).List(c.selector)
		if err != nil {
			klog.Errorf("Failed to list deployments: %v", err)
			return
		}
		for _, deployment := range deployments {
			c.kubeClient.AppsV1().Deployments(curService.Namespace).Delete(context.TODO(), deployment.Name, metav1.DeleteOptions{})
		}

		c.createEdgeWorker(curService, c.bigModelHost, BigModelPort)
		c.createCloudWorker(curService, BigModelPort)

		// update the service status
		c.client.JointInferenceServices(curService.Namespace).UpdateStatus(context.TODO(), curService, metav1.UpdateOptions{})
	}
	c.enqueueController(curService, true)
	c.syncToEdge(watch.Added, curService)
}
