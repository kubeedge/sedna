/*
Copyright 2025 The KubeEdge Authors.

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

package llmjointinference

import (
	"fmt"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
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

	sednav1 "github.com/kubeedge/sedna/pkg/apis/sedna/v1alpha1"
	sednaclientset "github.com/kubeedge/sedna/pkg/client/clientset/versioned/typed/sedna/v1alpha1"
	sednav1listers "github.com/kubeedge/sedna/pkg/client/listers/sedna/v1alpha1"
	"github.com/kubeedge/sedna/pkg/globalmanager/config"
	"github.com/kubeedge/sedna/pkg/globalmanager/runtime"
)

const (
	// Name is this controller name
	Name = "LLMJointInference"

	// KindName is the kind name of CR this controller controls
	KindName = "LLMJointInferenceService"
)

const (
	// Default ports for Triton inference server
	TritonHTTPPort    = 8000
	TritonGRPCPort    = 8001
	TritonMetricsPort = 8002
)

// gvk contains the schema.GroupVersionKind for this controller type.
var gvk = sednav1.SchemeGroupVersion.WithKind(KindName)

// Controller ensures that all LLMJointInferenceService objects
// have corresponding deployments/statefulsets to run their configured workload.
type Controller struct {
	kubeClient kubernetes.Interface
	client     sednaclientset.SednaV1alpha1Interface

	// deploymentsSynced returns true if the deployment store has been synced at least once.
	deploymentsSynced cache.InformerSynced
	// A store of deployments
	deploymentsLister appslisters.DeploymentLister

	// serviceStoreSynced returns true if the LLMJointInferenceService store has been synced at least once.
	serviceStoreSynced cache.InformerSynced
	// A store of services
	serviceLister sednav1listers.LLMJointInferenceServiceLister

	// LLMJointInferenceServices that need to be updated
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

func (c *Controller) enqueueController(obj interface{}, immediate bool) {
	key, err := cache.DeletionHandlingMetaNamespaceKeyFunc(obj)
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

	klog.Warningf("Error syncing LLM joint inference service: %v", err)
	c.queue.AddRateLimited(key)

	return true
}

// sync will sync the LLMJointInferenceService with the given key.
func (c *Controller) sync(key string) (bool, error) {
	startTime := time.Now()
	defer func() {
		klog.V(4).Infof("Finished syncing LLM joint inference service %q (%v)", key, time.Since(startTime))
	}()

	ns, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return false, err
	}

	service, err := c.serviceLister.LLMJointInferenceServices(ns).Get(name)
	if err != nil {
		if errors.IsNotFound(err) {
			klog.V(4).Infof("LLMJointInferenceService has been deleted: %v", key)
			return true, nil
		}
		return false, err
	}

	if isServiceFinished(service) {
		return true, nil
	}

	// Set kind for service
	service.SetGroupVersionKind(gvk)

	// Handle different deployment modes
	switch service.Spec.DeploymentMode {
	case sednav1.DeploymentModeSdk:
		// Use existing joint inference logic
		return c.handleSDKMode(service)
	case sednav1.DeploymentModeOneClick:
		// Handle new inference engine mode
		return c.handleOneClickMode(service)
	default:
		err = fmt.Errorf("unknown deployment mode: %s", service.Spec.DeploymentMode)
		c.recorder.Event(service, v1.EventTypeWarning, "InvalidDeploymentMode", err.Error())
		return false, err
	}
}

func (c *Controller) handleSDKMode(service *sednav1.LLMJointInferenceService) (bool, error) {
	// TODO: Implement SDK mode handling similar to existing joint inference controller
	return false, nil
}

func (c *Controller) handleOneClickMode(service *sednav1.LLMJointInferenceService) (bool, error) {
	switch service.Spec.InferenceConfig.Engine {
	case sednav1.Triton:
		return c.handleTritonEngine(service)
	case sednav1.VLLM:
		// TODO: Implement vLLM engine handling
		return false, fmt.Errorf("vLLM engine not implemented yet")
	case sednav1.SGLang:
		// TODO: Implement SGLang engine handling
		return false, fmt.Errorf("SGLang engine not implemented yet")
	default:
		err := fmt.Errorf("unknown inference engine: %s", service.Spec.InferenceConfig.Engine)
		c.recorder.Event(service, v1.EventTypeWarning, "InvalidInferenceEngine", err.Error())
		return false, err
	}
}

func (c *Controller) handleTritonEngine(service *sednav1.LLMJointInferenceService) (bool, error) {
	// Create or update Deployment for Triton server
	if err := c.syncTritonDeployment(service); err != nil {
		return false, err
	}

	// Create or update Service for Triton server
	if err := c.syncTritonService(service); err != nil {
		return false, err
	}

	// Update status
	if err := c.updateStatus(service); err != nil {
		return false, err
	}

	return true, nil
}

func isServiceFinished(service *sednav1.LLMJointInferenceService) bool {
	for _, cond := range service.Status.Conditions {
		if cond.Type == sednav1.LLMJointInferenceServiceCondFailed {
			return true
		}
	}
	return false
}

// New creates a new LLMJointInferenceService controller
func New(cc *runtime.ControllerContext) (runtime.FeatureControllerI, error) {
	deploymentInformer := cc.KubeInformerFactory.Apps().V1().Deployments()
	serviceInformer := cc.SednaInformerFactory.Sedna().V1alpha1().LLMJointInferenceServices()
	if cc.Config == nil {
		return nil, fmt.Errorf("controller config is required")
	}
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartStructuredLogging(0)
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{
		Interface: cc.KubeClient.CoreV1().Events(""),
	})

	c := &Controller{
		kubeClient:        cc.KubeClient,
		client:            cc.SednaClient.SednaV1alpha1(),
		deploymentsLister: deploymentInformer.Lister(),
		deploymentsSynced: deploymentInformer.Informer().HasSynced,

		serviceLister:      serviceInformer.Lister(),
		serviceStoreSynced: serviceInformer.Informer().HasSynced,
		queue: workqueue.NewRateLimitingQueueWithConfig(
			workqueue.NewItemExponentialFailureRateLimiter(runtime.DefaultBackOff, runtime.MaxBackOff),
			workqueue.RateLimitingQueueConfig{Name: strings.ToLower(Name)},
		),
		recorder: eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: strings.ToLower(Name) + "-controller"}),
		cfg:      cc.Config,
	}

	// Set up event handlers for LLMJointInferenceService
	serviceInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			c.enqueueController(obj, false)
			c.syncToEdge(watch.Added, obj)
		},
		UpdateFunc: func(old, new interface{}) {
			c.enqueueController(new, false)
			c.syncToEdge(watch.Modified, new)
		},
		DeleteFunc: func(obj interface{}) {
			c.enqueueController(obj, false)
			c.syncToEdge(watch.Deleted, obj)
		},
	})

	// Set up event handlers for Deployment
	deploymentInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    c.addDeployment,
		UpdateFunc: c.updateDeployment,
		DeleteFunc: c.deleteDeployment,
	})

	return c, nil
}
