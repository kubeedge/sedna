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
	"strings"
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

// gvk contains the schema.GroupVersionKind for this controller type.
var gvk = sednav1.SchemeGroupVersion.WithKind(KindName)

// Controller ensures that all JointInferenceService objects
// have corresponding deployments to run their configured workload.
type Controller struct {
	kubeClient kubernetes.Interface
	client     sednaclientset.SednaV1alpha1Interface

	// deploymentsSynced returns true if the deployment store has been synced at least once.
	deploymentsSynced cache.InformerSynced
	// A store of deployment
	deploymentsLister appslisters.DeploymentLister

	// serviceStoreSynced returns true if the JointInferenceService store has been synced at least once.
	serviceStoreSynced cache.InformerSynced
	// A store of service
	serviceLister sednav1listers.JointInferenceServiceLister

	// JointInferenceServices that need to be updated
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

	if !cache.WaitForNamedCacheSync(Name, stopCh, c.deploymentsSynced, c.serviceStoreSynced) {
		klog.Errorf("failed to wait for %s caches to sync", Name)

		return
	}

	klog.Infof("Starting %s workers", Name)
	for i := 0; i < workers; i++ {
		go wait.Until(c.worker, time.Second, stopCh)
	}

	<-stopCh
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

	// Use Lister to obtain the JointInferenceService object (Lister is a cache reading mechanism).
	// If the service does not exist (has been deleted), log the message and return true, indicating that this object no longer needs to be synchronized.
	// If the acquisition fails but not because the object has been deleted, return an error.
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
	service.SetGroupVersionKind(gvk)

	selectorDeployments, _ := runtime.GenerateSelector(&service)
	deployments, err := c.deploymentsLister.Deployments(service.Namespace).List(selectorDeployments)

	if err != nil {
		return false, err
	}

	klog.V(4).Infof("list jointinference service %v/%v, %v deployments: %v", service.Namespace, service.Name, len(deployments), deployments)

	latestConditionLen := len(service.Status.Conditions)

	activeDeployments := runtime.CalcActiveDeploymentCount(deployments)
	var failed int32 = 0

	// neededCounts means that two deployments should be created successfully in a jointinference service currently
	// two deployments consist of edge deployment and cloud deployment
	var neededCounts int32 = 2

	if service.Status.StartTime == nil {
		now := metav1.Now()
		service.Status.StartTime = &now
	} else {
		failed = neededCounts - activeDeployments
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

	if failed > 0 {
		serviceFailed = true
		// TODO: get the failed worker, and knows that which worker fails, edge inference worker or cloud inference worker
		reason = "workerFailed"
		message = "the worker of service failed"
		newCondtionType = sednav1.JointInferenceServiceCondFailed
		c.recorder.Event(&service, v1.EventTypeWarning, reason, message)
	} else {
		if len(deployments) == 0 {
			activeDeployments, manageServiceErr = c.createWorkers(&service)
		}
		if manageServiceErr != nil {
			serviceFailed = true
			message = error.Error(manageServiceErr)
			newCondtionType = sednav1.JointInferenceServiceCondFailed
			failed = neededCounts - activeDeployments
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

	// no need to update the jointinferenceservice if the status hasn't changed since last time
	if service.Status.Active != activeDeployments || service.Status.Failed != failed || len(service.Status.Conditions) != latestConditionLen {
		service.Status.Active = activeDeployments
		service.Status.Failed = failed

		if err := c.updateStatus(&service); err != nil {
			return forget, err
		}

		if serviceFailed && !isServiceFinished(&service) {
			// returning an error will re-enqueue jointinferenceservice after the backoff period
			return forget, fmt.Errorf("failed deployment(s) detected for jointinference service key %q", key)
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

func (c *Controller) createWorkers(service *sednav1.JointInferenceService) (active int32, err error) {
	active = 0

	var bigModelPort int32 = BigModelPort
	// create cloud worker
	err = c.createCloudWorker(service, bigModelPort)
	if err != nil {
		return active, err
	}
	active++

	// create k8s service for cloud deployment
	bigModelHost, err := runtime.CreateEdgeMeshService(c.kubeClient, service, jointInferenceForCloud, bigModelPort)
	if err != nil {
		return active, err
	}

	// create edge worker
	err = c.createEdgeWorker(service, bigModelHost, bigModelPort)
	if err != nil {
		return active, err
	}
	active++

	return active, err
}

// enqueueByDeployment enqueues the JointInferenceService object of the specified deployment.
func (c *Controller) enqueueByDeployment(deployment *appsv1.Deployment, immediate bool) {
	controllerRef := metav1.GetControllerOf(deployment)

	klog.Infof("Deployment enqueued %v", deployment.Kind)

	if controllerRef == nil {
		return
	}

	if controllerRef.Kind != gvk.Kind {
		return
	}

	service, err := c.serviceLister.JointInferenceServices(deployment.Namespace).Get(controllerRef.Name)
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

// When a deployment is updated, figure out what jointinferenceservice manage it and wake them up.
func (c *Controller) updateDeployment(old, cur interface{}) {
	oldD := old.(*appsv1.Deployment)
	curD := cur.(*appsv1.Deployment)
	// no deployment update, no queue
	if curD.ResourceVersion == oldD.ResourceVersion {
		return
	}

	c.addDeployment(curD)
}

// deleteDeployment enqueues the jointinferenceservice obj When a deleteDeployment is deleted
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

	// If the deployment is accidentally deleted, recreate the deployment.
	newDeployment := deployment.DeepCopy()
	serviceName := func(input string) string {
		return strings.Split(input, "-deployment")[0]
	}(newDeployment.Name)
	_, err := c.serviceLister.JointInferenceServices(newDeployment.Namespace).Get(serviceName)
	if !errors.IsNotFound(err) {
		// Remove unnecessary metadata.
		newDeployment.ResourceVersion = ""
		newDeployment.UID = ""
		// Create a new deployment.
		_, err := c.kubeClient.AppsV1().Deployments(newDeployment.Namespace).Create(context.TODO(), newDeployment, metav1.CreateOptions{})
		if err != nil {
			klog.Errorf("failed to recreate deployment %s: %v", deployment.Name, err)
			return
		}
	}

	klog.Infof("Successfully recreated deployment %s", deployment.Name)
	c.enqueueByDeployment(newDeployment, true)
}

func (c *Controller) updateInferenceServices(old, cur interface{}) error {
	oldService := old.(*sednav1.JointInferenceService)
	newService := cur.(*sednav1.JointInferenceService)
	// Check the changes in specific fields and perform corresponding operations.
	if !reflect.DeepEqual(oldService.Spec.CloudWorker, newService.Spec.CloudWorker) {
		// If the cloud inference service changes, perform the corresponding update operation.
		return c.updateCloudWorker(newService)
	}

	// Obtain the address of the cloud inference service.
	var bigModelHost string
	svc, err := c.kubeClient.CoreV1().Services(oldService.Namespace).Get(context.Background(),
		strings.ToLower(oldService.Name+"-"+jointInferenceForCloud), metav1.GetOptions{})
	if err != nil {
		if errors.IsNotFound(err) {
			bigModelHost, err = runtime.CreateEdgeMeshService(c.kubeClient, oldService, jointInferenceForCloud, BigModelPort)
			if err != nil {
				return err
			}
		}
	}
	bigModelHost = fmt.Sprintf("%s.%s", svc.Name, svc.Namespace)

	if !reflect.DeepEqual(oldService.Spec.EdgeWorker, newService.Spec.EdgeWorker) {
		// If the edge inference service changes, perform the corresponding update operation.
		return c.updateEdgeWorker(newService, bigModelHost)
	}

	return nil
}

func (c *Controller) createOrUpdateWorker(service *sednav1.JointInferenceService, workerType string, bigModelHost string, bigModelPort int32, create bool) error {
	var modelName string
	var modelTemplate v1.PodTemplateSpec
	var workerParam runtime.WorkerParam

	// Set the corresponding parameters according to the workerType.
	switch workerType {
	case jointInferenceForCloud:
		modelName = service.Spec.CloudWorker.Model.Name
		modelTemplate = *service.Spec.CloudWorker.Template.DeepCopy()

		workerParam.Env = map[string]string{
			"BIG_MODEL_BIND_PORT": strconv.Itoa(int(bigModelPort)),
		}
		workerParam.WorkerType = workerType
		workerParam.HostNetwork = false // The cloud does not need HostNetwork.

	case jointInferenceForEdge:
		modelName = service.Spec.EdgeWorker.Model.Name
		modelTemplate = *service.Spec.EdgeWorker.Template.DeepCopy()

		HEMParameterJSON, _ := json.Marshal(service.Spec.EdgeWorker.HardExampleMining.Parameters)
		HEMParameterString := string(HEMParameterJSON)

		workerParam.Env = map[string]string{
			"BIG_MODEL_IP":   bigModelHost,
			"BIG_MODEL_PORT": strconv.Itoa(int(bigModelPort)),
			"HEM_NAME":       service.Spec.EdgeWorker.HardExampleMining.Name,
			"HEM_PARAMETERS": HEMParameterString,

			"LC_SERVER": c.cfg.LC.Server,
		}
		workerParam.WorkerType = workerType
		workerParam.HostNetwork = true // Edge nodes need HostNetwork.
	}

	// get the model.
	model, err := c.client.Models(service.Namespace).Get(context.Background(), modelName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get model %s: %w", modelName, err)
	}

	secretName := model.Spec.CredentialName
	var modelSecret *v1.Secret
	if secretName != "" {
		modelSecret, _ = c.kubeClient.CoreV1().Secrets(service.Namespace).Get(context.TODO(), secretName, metav1.GetOptions{})
	}

	// Fill in the mounting configuration of workerParam.
	workerParam.Mounts = append(workerParam.Mounts, runtime.WorkerMount{
		URL: &runtime.MountURL{
			URL:                   model.Spec.URL,
			Secret:                modelSecret,
			DownloadByInitializer: true,
		},
		Name:    "model",
		EnvName: "MODEL_URL",
	})

	// Set other common environment variables.
	workerParam.Env["NAMESPACE"] = service.Namespace
	workerParam.Env["SERVICE_NAME"] = service.Name
	workerParam.Env["WORKER_NAME"] = strings.ToLower(workerType) + "worker-" + utilrand.String(5)

	// Create or update Deployment.
	if create {
		_, err = runtime.CreateDeploymentWithTemplate(c.kubeClient, service, &appsv1.DeploymentSpec{Template: modelTemplate}, &workerParam)
	} else {
		service.SetGroupVersionKind(gvk)
		workerName := service.Name + "-deployment-" + strings.ToLower(workerType)
		existingDeployment, err := c.deploymentsLister.Deployments(service.Namespace).Get(workerName)
		if err != nil {
			return fmt.Errorf("get %s Deployment failed:%v", strings.ToLower(workerType), err)
		}
		newDeployment := existingDeployment.DeepCopy()
		newDeployment.Spec.Template = modelTemplate
		_, err = runtime.UpdateDeploymentWithTemplate(c.kubeClient, service, newDeployment, &workerParam)
	}
	return err
}

func (c *Controller) createCloudWorker(service *sednav1.JointInferenceService, bigModelPort int32) error {
	return c.createOrUpdateWorker(service, jointInferenceForCloud, "", bigModelPort, true)
}

func (c *Controller) createEdgeWorker(service *sednav1.JointInferenceService, bigModelHost string, bigModelPort int32) error {
	return c.createOrUpdateWorker(service, jointInferenceForEdge, bigModelHost, bigModelPort, true)
}

func (c *Controller) updateCloudWorker(newservice *sednav1.JointInferenceService) error {
	return c.createOrUpdateWorker(newservice, jointInferenceForCloud, "", BigModelPort, false)
}

func (c *Controller) updateEdgeWorker(newservice *sednav1.JointInferenceService, bigModelHost string) error {
	return c.createOrUpdateWorker(newservice, jointInferenceForEdge, bigModelHost, BigModelPort, false)
}

// New creates a new JointInferenceService controller that keeps the relevant deployments
// in sync with their corresponding JointInferenceService objects.
func New(cc *runtime.ControllerContext) (runtime.FeatureControllerI, error) {
	cfg := cc.Config

	deploymentInformer := cc.KubeInformerFactory.Apps().V1().Deployments()

	serviceInformer := cc.SednaInformerFactory.Sedna().V1alpha1().JointInferenceServices()

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

		UpdateFunc: func(old, cur interface{}) {
			jc.enqueueController(cur, true)
			jc.updateInferenceServices(old, cur)
			jc.syncToEdge(watch.Modified, cur)
		},

		DeleteFunc: func(obj interface{}) {
			jc.enqueueController(obj, true)
			jc.syncToEdge(watch.Deleted, obj)
		},
	})

	jc.serviceLister = serviceInformer.Lister()
	jc.serviceStoreSynced = serviceInformer.Informer().HasSynced

	deploymentInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    jc.addDeployment,
		UpdateFunc: jc.updateDeployment,
		DeleteFunc: jc.deleteDeployment,
	})
	jc.deploymentsLister = deploymentInformer.Lister()
	jc.deploymentsSynced = deploymentInformer.Informer().HasSynced

	return jc, nil
}
