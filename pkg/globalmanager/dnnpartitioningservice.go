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
	dnnPartitioningForEdge  = "edge"
	dnnPartitioningForCloud = "cloud"
)

// dnnPartitioningServiceControllerKind contains the schema.GroupVersionKind for this controller type.
var dnnPartitioningServiceControllerKind = sednav1.SchemeGroupVersion.WithKind("DnnPartitioningService")

// DnnPartitioningServiceController ensures that all DnnPartitioningService objects
// have corresponding pods to run their configured workload.
type DnnPartitioningServiceController struct {
	kubeClient kubernetes.Interface
	client     sednaclientset.SednaV1alpha1Interface

	// podStoreSynced returns true if the pod store has been synced at least once.
	podStoreSynced cache.InformerSynced
	// A store of pods
	podStore corelisters.PodLister

	// serviceStoreSynced returns true if the dnnpartitioningservice store has been synced at least once.
	serviceStoreSynced cache.InformerSynced
	// A store of service
	serviceLister sednav1listers.DNNPartitioningServiceLister

	// deploymentsSynced returns true if the deployment store has been synced at least once.
	deploymentsSynced cache.InformerSynced
	// A store of deployment
	deploymentsLister appslisters.DeploymentLister

	enqueueDeployment func(deployment *appsv1.Deployment)

		
	// DnnPartitioningServices that need to be updated
	queue workqueue.RateLimitingInterface

	recorder record.EventRecorder

	cfg *config.ControllerConfig
}

// Start starts the main goroutine responsible for watching and syncing services.
func (dc *DnnPartitioningServiceController) Start() error {
	workers := 1
	stopCh := messageContext.Done()

	go func() {
		defer utilruntime.HandleCrash()
		defer dc.queue.ShutDown()
		klog.Infof("Starting dnn partitioning service controller..")
		defer klog.Infof("Shutting down dnn partitioning service controller")

		if !cache.WaitForNamedCacheSync("dnnpartitioningservice", stopCh, dc.podStoreSynced, dc.serviceStoreSynced) {
			klog.Errorf("failed to wait for dnn partitioning service caches to sync")

			return
		}

		klog.Infof("Starting dnn partitioning service workers")
		for i := 0; i < workers; i++ {
			go wait.Until(dc.worker, time.Second, stopCh)
		}

		<-stopCh
	}()
	return nil
}

/*

DEPLOYMENT HOOKS

*/

func (dc *DnnPartitioningServiceController) enqueueByDeployment(deployment *appsv1.Deployment) {
	controllerRef := metav1.GetControllerOf(deployment)

	klog.Infof("Deployment enqueued %v", deployment.Kind)

	if controllerRef == nil {
		return
	}

	if controllerRef.Kind != MultiEdgeTrackingServiceKind.Kind {
		return
	}

	service, err := dc.serviceLister.DNNPartitioningServices(deployment.Namespace).Get(controllerRef.Name)
	if err != nil {
		return
	}

	if service.UID != controllerRef.UID {
		return
	}

	dc.enqueueController(service, true)
}

func (dc *DnnPartitioningServiceController) addDeployment(obj interface{}) {
	deployment := obj.(*appsv1.Deployment)
	dc.enqueueByDeployment(deployment)
}

//deleteDeployment enqueues the ObjectSearchService obj When a deleteDeployment is deleted
func (dc *DnnPartitioningServiceController) deleteDeployment(obj interface{}) {
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
	dc.enqueueByDeployment(deployment)
}

// When a deployment is updated, figure out what object search service manage it and wake them up.
func (dc *DnnPartitioningServiceController) updateDeployment(old, cur interface{}) {
	oldD := old.(*appsv1.Deployment)
	curD := cur.(*appsv1.Deployment)
	// no deployment update, no queue
	if curD.ResourceVersion == oldD.ResourceVersion {
		return
	}

	dc.addDeployment(curD)
}

// enqueueByPod enqueues the dnnPartitioningService object of the specified pod.
func (dc *DnnPartitioningServiceController) enqueueByPod(pod *v1.Pod, immediate bool) {
	controllerRef := metav1.GetControllerOf(pod)

	if controllerRef == nil {
		return
	}

	if controllerRef.Kind != dnnPartitioningServiceControllerKind.Kind {
		return
	}

	service, err := dc.serviceLister.DNNPartitioningServices(pod.Namespace).Get(controllerRef.Name)
	if err != nil {
		return
	}

	if service.UID != controllerRef.UID {
		return
	}

	dc.enqueueController(service, immediate)
}

// When a pod is created, enqueue the controller that manages it and update it's expectations.
func (dc *DnnPartitioningServiceController) addPod(obj interface{}) {
	pod := obj.(*v1.Pod)
	if pod.DeletionTimestamp != nil {
		// on a restart of the controller, it's possible a new pod shows up in a state that
		// is already pending deletion. Prevent the pod from being a creation observation.
		dc.deletePod(pod)
		return
	}

	// backoff to queue when PodFailed
	immediate := pod.Status.Phase != v1.PodFailed

	dc.enqueueByPod(pod, immediate)
}

// When a pod is updated, figure out what dnn partitioning service manage it and wake them up.
func (dc *DnnPartitioningServiceController) updatePod(old, cur interface{}) {
	curPod := cur.(*v1.Pod)
	oldPod := old.(*v1.Pod)

	// no pod update, no queue
	if curPod.ResourceVersion == oldPod.ResourceVersion {
		return
	}

	dc.addPod(curPod)
}

// deletePod enqueues the dnnPartitioningservice obj When a pod is deleted
func (dc *DnnPartitioningServiceController) deletePod(obj interface{}) {
	pod, ok := obj.(*v1.Pod)

	// comment from https://github.com/kubernetes/kubernetes/blob/master/pkg/controller/job/job_controller.go

	// When a delete is dropped, the relist will notice a pod in the store not
	// in the list, leading to the insertion of a tombstone object which contains
	// the deleted key/value. Note that this value might be stale. If the pod
	// changed labels the new dnnpartitioningservice will not be woken up till the periodic resync.
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
	dc.enqueueByPod(pod, true)
}

// obj could be an *sednav1.DnnPartitioningService, or a DeletionFinalStateUnknown marker item,
// immediate tells the controller to update the status right away, and should
// happen ONLY when there was a successful pod run.
func (dc *DnnPartitioningServiceController) enqueueController(obj interface{}, immediate bool) {
	key, err := k8scontroller.KeyFunc(obj)
	if err != nil {
		klog.Warningf("Couldn't get key for object %+v: %v", obj, err)
		return
	}

	backoff := time.Duration(0)
	if !immediate {
		backoff = getBackoff(dc.queue, key)
	}
	dc.queue.AddAfter(key, backoff)
}

// worker runs a worker thread that just dequeues items, processes them, and marks them done.
// It enforces that the sync is never invoked concurrently with the same key.
func (dc *DnnPartitioningServiceController) worker() {
	for dc.processNextWorkItem() {
	}
}

func (dc *DnnPartitioningServiceController) processNextWorkItem() bool {
	key, quit := dc.queue.Get()
	if quit {
		return false
	}
	defer dc.queue.Done(key)

	forget, err := dc.sync(key.(string))
	if err == nil {
		if forget {
			dc.queue.Forget(key)
		}
		return true
	}

	klog.Warningf("Error syncing dnnpartitioning service: %v", err)
	dc.queue.AddRateLimited(key)

	return true
}

// sync will sync the dnnpartitioningservice with the given key.
// This function is not meant to be invoked concurrently with the same key.
func (dc *DnnPartitioningServiceController) sync(key string) (bool, error) {
	startTime := time.Now()
	defer func() {
		klog.V(4).Infof("Finished syncing dnnpartitioning service %q (%v)", key, time.Since(startTime))
	}()

	ns, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return false, err
	}
	if len(ns) == 0 || len(name) == 0 {
		return false, fmt.Errorf("invalid dnnpartitioning service key %q: either namespace or name is missing", key)
	}
	sharedDnnpartitioningservice, err := dc.serviceLister.DNNPartitioningServices(ns).Get(name)
	if err != nil {
		if errors.IsNotFound(err) {
			klog.V(4).Infof("DnnPartitioningService has been deleted: %v", key)
			return true, nil
		}
		return false, err
	}

	dnnpartitioningservice := *sharedDnnpartitioningservice

	// if dnnpartitioningservice was finished previously, we don't want to redo the termination
	if isDnnpartitioningserviceFinished(&dnnpartitioningservice) {
		return true, nil
	}

	// set kind for dnnpartitioningservice in case that the kind is None
	// more details at https://github.com/kubernetes/kubernetes/issues/3030
	dnnpartitioningservice.SetGroupVersionKind(dnnPartitioningServiceControllerKind)

	selector, _ := GenerateSelector(&dnnpartitioningservice)
	pods, err := dc.podStore.Pods(dnnpartitioningservice.Namespace).List(selector)
	deployment, err := dc.deploymentsLister.Deployments(dnnpartitioningservice.Namespace).List(selector)

	if err != nil {
		return false, err
	}

	klog.V(4).Infof("list dnnpartitioning service %v/%v, %v pods: %v", dnnpartitioningservice.Namespace, dnnpartitioningservice.Name, len(pods), pods)

	latestConditionLen := len(dnnpartitioningservice.Status.Conditions)

	activePods := calcActivePodCount(pods)
	activeDeployments := calcActiveDeploymentCount(deployment)
	
	var failedPods, failedDeployment int32 = 0, 0

	var neededPodCounts int32 = *sharedDnnpartitioningservice.Spec.DNNPartitioningEdgeWorker.Spec.Replicas +
		*sharedDnnpartitioningservice.Spec.DNNPartitioningCloudWorker.Spec.Replicas

	var neededDeploymentCounts int32 = int32(reflect.TypeOf(sednav1.DNNPartitioningServiceSpec{}).NumField())

	// first start
	if dnnpartitioningservice.Status.StartTime == nil {
		now := metav1.Now()
		dnnpartitioningservice.Status.StartTime = &now
	} else {
		failedPods = neededPodCounts - activePods
		failedDeployment = neededDeploymentCounts - activeDeployments
	}

	var manageServiceErr error
	serviceFailed := false

	var latestConditionType sednav1.DnnPartitioningServiceConditionType = ""

	// get the latest condition type
	// based on that condition updated is appended, not inserted.
	jobConditions := dnnpartitioningservice.Status.Conditions
	if len(jobConditions) > 0 {
		latestConditionType = (jobConditions)[len(jobConditions)-1].Type
	}

	var newCondtionType sednav1.DnnPartitioningServiceConditionType
	var reason string
	var message string

	if failedPods > 0 || failedDeployment > 0 {
		serviceFailed = true
		// TODO: Split code to handle deployment failure separately
		// TODO: get the failed worker, and knows that which worker fails, edge inference worker or cloud inference worker
		reason = "workerFailed"
		message = "the worker of dnnpartitioningservice failed"
		newCondtionType = sednav1.DNNPartitioningServiceCondFailed
		dc.recorder.Event(&dnnpartitioningservice, v1.EventTypeWarning, reason, message)
	} else {
		if len(pods) == 0 {
			activePods, activeDeployments, manageServiceErr = dc.createWorkers(&dnnpartitioningservice)
		}
		if manageServiceErr != nil {
			serviceFailed = true
			message = error.Error(manageServiceErr)
			newCondtionType = sednav1.DNNPartitioningServiceCondFailed
			failedPods = neededPodCounts - activePods
			failedDeployment = neededDeploymentCounts - activeDeployments
		} else {
			// TODO: handle the case that the pod phase is PodSucceeded
			newCondtionType = sednav1.DNNPartitioningServiceCondRunning
		}
	}

	//
	if newCondtionType != latestConditionType {
		dnnpartitioningservice.Status.Conditions = append(dnnpartitioningservice.Status.Conditions, NewDnnPartitioningServiceCondition(newCondtionType, reason, message))
	}
	forget := false

	// no need to update the dnnpartitioningservice if the status hasn't changed since last time
	if dnnpartitioningservice.Status.Active != activePods || dnnpartitioningservice.Status.Failed != failedPods || len(dnnpartitioningservice.Status.Conditions) != latestConditionLen {
		dnnpartitioningservice.Status.Active = activePods
		dnnpartitioningservice.Status.Failed = failedPods

		if err := dc.updateStatus(&dnnpartitioningservice); err != nil {
			return forget, err
		}

		if serviceFailed && !isDnnpartitioningserviceFinished(&dnnpartitioningservice) {
			// returning an error will re-enqueue dnnpartitioningservice after the backoff period
			return forget, fmt.Errorf("failed pod(s) detected for dnnpartitioning service key %q", key)
		}

		forget = true
	}

	return forget, manageServiceErr
}

// NewDNNPartitioningServiceCondition creates a new DNNPartitioning condition
func NewDnnPartitioningServiceCondition(conditionType sednav1.DnnPartitioningServiceConditionType, reason, message string) sednav1.DNNPartitioningServiceCondition {
	return sednav1.DNNPartitioningServiceCondition{
		Type:               conditionType,
		Status:             v1.ConditionTrue,
		LastHeartbeatTime:  metav1.Now(),
		LastTransitionTime: metav1.Now(),
		Reason:             reason,
		Message:            message,
	}
}

func (dc *DnnPartitioningServiceController) updateStatus(dnnpartitioningservice *sednav1.DNNPartitioningService) error {
	serviceClient := dc.client.DNNPartitioningServices(dnnpartitioningservice.Namespace)
	var err error
	for i := 0; i <= ResourceUpdateRetries; i = i + 1 {
		var newDnnpartitioningservice *sednav1.DNNPartitioningService
		newDnnpartitioningservice, err = serviceClient.Get(context.TODO(), dnnpartitioningservice.Name, metav1.GetOptions{})
		if err != nil {
			break
		}
		newDnnpartitioningservice.Status = dnnpartitioningservice.Status
		if _, err = serviceClient.UpdateStatus(context.TODO(), newDnnpartitioningservice, metav1.UpdateOptions{}); err == nil {
			break
		}
	}
	return nil
}

func isDnnpartitioningserviceFinished(j *sednav1.DNNPartitioningService) bool {
	for _, c := range j.Status.Conditions {
		if (c.Type == sednav1.DNNPartitioningServiceCondFailed) && c.Status == v1.ConditionTrue {
			return true
		}
	}
	return false
}

func (dc *DnnPartitioningServiceController) createWorkers(service *sednav1.DNNPartitioningService) (activePods int32, activeDeployments int32, err error) {
	activePods = 0
	activeDeployments = 0

		// create k8s service for cloudPod
	// FIXME(llhuii): only the case that Spec.NodeName specified is support,
	// will support Spec.NodeSelector.
	cloudModelIP, err := GetNodeIPByName(dc.kubeClient, service.Spec.DNNPartitioningCloudWorker.Spec.Template.Spec.NodeName)
	cloudServicePort, err := CreateKubernetesService(dc.kubeClient, service, dnnPartitioningForCloud, cloudModelPort, cloudModelIP)

	// create cloud worker
	err = dc.createDNNPartitioningCloudWorker(service, cloudServicePort)
	if err != nil {
		return activePods, activeDeployments, err
	}
	activeDeployments++


	// create edge worker
	err = dc.createDNNPartitioningEdgeWorker(service, cloudServicePort)
	if err != nil {
		return activePods, activeDeployments, err
	}
	activeDeployments++

	return activePods, activeDeployments, err
}

func (dc *DnnPartitioningServiceController) createDNNPartitioningCloudWorker(service *sednav1.DNNPartitioningService, port int32) error {
	// deliver pod for cloudworker
	cloudModelName := service.Spec.DNNPartitioningCloudWorker.Model.Name
	cloudModel, err := dc.client.Models(service.Namespace).Get(context.Background(), cloudModelName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get cloud model %s: %w",
			cloudModelName, err)
	}

	var workerParam WorkerParam

	secretName := cloudModel.Spec.CredentialName
	var modelSecret *v1.Secret
	if secretName != "" {
		modelSecret, _ = dc.kubeClient.CoreV1().Secrets(service.Namespace).Get(context.TODO(), secretName, metav1.GetOptions{})
	}
	workerParam.mounts = append(workerParam.mounts, WorkerMount{
		URL: &MountURL{
			URL:                   cloudModel.Spec.URL,
			Secret:                modelSecret,
			DownloadByInitializer: true,
		},
		Name:    "model",
		EnvName: "MODEL_URL",
	})

	workerParam.env = map[string]string{
		"NAMESPACE":    service.Namespace,
		"SERVICE_NAME": service.Name,
		"WORKER_NAME":  "cloudworker-" + utilrand.String(5),

		"CLOUD_MODEL_BIND_PORT": strconv.Itoa(int(cloudModelPort)),
	}

	workerParam.workerType = dnnPartitioningForCloud

	// create cloud pod
	// _, err = createPodWithTemplate(dc.kubeClient,
	// 	service,
	// 	&service.Spec.DNNPartitioningCloudWorker.Template,
	// 	&workerParam)

		// Create OD deployment AND related pods (as part of the deployment creation)
	_, err = CreateDeploymentWithTemplate(dc.kubeClient, service, &service.Spec.DNNPartitioningCloudWorker.Spec, &workerParam, port)
	// if err != nil {
	// 	return activePods, activeDeployments, fmt.Errorf("failed to create reid workers deployment: %w", err)
	// }
	// //activeDeployments++

	return err
}

func (dc *DnnPartitioningServiceController) createDNNPartitioningEdgeWorker(service *sednav1.DNNPartitioningService, cloudServicePort int32) error {
	// deliver pod for edgeworker
	ctx := context.Background()
	edgeWorker := service.Spec.DNNPartitioningEdgeWorker
	var perr error

	edgeModelName := edgeWorker.Model.Name
	edgeModel, err := dc.client.Models(service.Namespace).Get(ctx, edgeModelName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get edge model %s: %w",
			edgeModelName, err)
	}

	secretName := edgeModel.Spec.CredentialName
	var modelSecret *v1.Secret
	if secretName != "" {
		modelSecret, _ = dc.kubeClient.CoreV1().Secrets(service.Namespace).Get(context.TODO(), secretName, metav1.GetOptions{})
	}

	// FIXME(llhuii): only the case that Spec.NodeName specified is support,
	// will support Spec.NodeSelector.
	// get cloudModelIP from nodeName in cloudWorker
	cloudModelIP, err := GetNodeIPByName(dc.kubeClient, service.Spec.DNNPartitioningCloudWorker.Spec.Template.Spec.NodeName)
	if err != nil {
		return fmt.Errorf("failed to get node ip: %w", err)
	}

	//HEMParameterJSON, _ := json.Marshal(edgeWorker.HardExampleMining.Parameters)
	//HEMParameterString := string(HEMParameterJSON)
	var workerParam WorkerParam

	workerParam.mounts = append(workerParam.mounts, WorkerMount{
		URL: &MountURL{
			URL:                   edgeModel.Spec.URL,
			Secret:                modelSecret,
			DownloadByInitializer: true,
		},
		Name:    "model",
		EnvName: "MODEL_URL",
	})

	workerParam.env = map[string]string{
		"NAMESPACE":    service.Namespace,
		"SERVICE_NAME": service.Name,
		"WORKER_NAME":  "edgeworker-" + utilrand.String(5),

		"CLOUD_MODEL_IP":   cloudModelIP,
		"CLOUD_MODEL_PORT": strconv.Itoa(int(cloudServicePort)),

		//"HEM_NAME":       edgeWorker.HardExampleMining.Name,
		//"HEM_PARAMETERS": HEMParameterString,

		"LC_SERVER": dc.cfg.LC.Server,
	}

	workerParam.workerType = dnnPartitioningForEdge
	workerParam.hostNetwork = true

	// create edge pod
	// _, perr = createPodWithTemplate(dc.kubeClient,
	// 	service,
	// 	&edgeNode.Template,
	// 	&workerParam)

	_, perr = CreateDeploymentWithTemplate(dc.kubeClient, service, &edgeWorker.Spec, &workerParam, 8000)
	// if err != nil {
	// 	return activePods, activeDeployments, fmt.Errorf("failed to create reid workers deployment: %w", err)
	// }



	return perr
}

// GetName returns the name of the DNNPartitioning inference controller
func (dc *DnnPartitioningServiceController) GetName() string {
	return "DnnPartitioningServiceController"
}

// NewDNNPartitioningController creates a new DNNPartitioningService controller that keeps the relevant pods
// in sync with their corresponding DNNPartitioningService objects.
func NewDNNPartitioningtController(cfg *config.ControllerConfig) (FeatureControllerI, error) {
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
	serviceInformer := serviceInformerFactory.Sedna().V1alpha1().DNNPartitioningServices()

	deploymentInformer := kubeInformerFactory.Apps().V1().Deployments()
	
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: kubeClient.CoreV1().Events("")})



	dc := &DnnPartitioningServiceController{
		kubeClient: kubeClient,
		client:     crdclient.SednaV1alpha1(),

		queue:    workqueue.NewNamedRateLimitingQueue(workqueue.NewItemExponentialFailureRateLimiter(DefaultBackOff, MaxBackOff), "dnnpartitioningservice"),
		recorder: eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "dnnpartitioningservice-controller"}),
		cfg:      cfg,
	}

	serviceInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			dc.enqueueController(obj, true)
		},

		UpdateFunc: func(old, cur interface{}) {
			dc.enqueueController(cur, true)
		},

		DeleteFunc: func(obj interface{}) {
			dc.enqueueController(obj, true)
		},
	})

	dc.serviceLister = serviceInformer.Lister()
	dc.serviceStoreSynced = serviceInformer.Informer().HasSynced

	podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    dc.addPod,
		UpdateFunc: dc.updatePod,
		DeleteFunc: dc.deletePod,
	})

	dc.podStore = podInformer.Lister()
	dc.podStoreSynced = podInformer.Informer().HasSynced

	deploymentInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    dc.addDeployment,
		UpdateFunc: dc.updateDeployment,
		DeleteFunc: dc.deleteDeployment,
	})

	dc.deploymentsLister = deploymentInformer.Lister()
	dc.deploymentsSynced = deploymentInformer.Informer().HasSynced

	stopCh := messageContext.Done()
	kubeInformerFactory.Start(stopCh)
	serviceInformerFactory.Start(stopCh)
	return dc, err
}
