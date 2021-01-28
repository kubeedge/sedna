package globalmanager

import (
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"
	"strconv"
	"strings"
	"time"

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

type jointInferenceType string

const (
	jointInferenceForEdge  jointInferenceType = "Edge"
	jointInferenceForCloud jointInferenceType = "Cloud"
)

// jointServiceControllerKind contains the schema.GroupVersionKind for this controller type.
var jointServiceControllerKind = sednav1.SchemeGroupVersion.WithKind("JointInferenceService")

// JointInferenceServiceController ensures that all JointInferenceService objects
// have corresponding pods to run their configured workload.
type JointInferenceServiceController struct {
	kubeClient kubernetes.Interface
	client     sednaclientset.SednaV1alpha1Interface

	// podStoreSynced returns true if the pod store has been synced at least once.
	podStoreSynced cache.InformerSynced
	// A store of pods
	podStore corelisters.PodLister

	// serviceStoreSynced returns true if the jointinferenceservice store has been synced at least once.
	serviceStoreSynced cache.InformerSynced
	// A store of service
	serviceLister sednav1listers.JointInferenceServiceLister

	// JointInferenceServices that need to be updated
	queue workqueue.RateLimitingInterface

	recorder record.EventRecorder

	cfg *config.ControllerConfig
}

// Start starts the main goroutine responsible for watching and syncing services.
func (jc *JointInferenceServiceController) Start() error {
	workers := 1
	stopCh := messageContext.Done()

	go func() {
		defer utilruntime.HandleCrash()
		defer jc.queue.ShutDown()
		klog.Infof("Starting joint inference service controller")
		defer klog.Infof("Shutting down joint inference service controller")

		if !cache.WaitForNamedCacheSync("jointinferenceservice", stopCh, jc.podStoreSynced, jc.serviceStoreSynced) {
			klog.Errorf("failed to wait for joint inferce service caches to sync")

			return
		}

		klog.Infof("Starting joint inference service workers")
		for i := 0; i < workers; i++ {
			go wait.Until(jc.worker, time.Second, stopCh)
		}

		<-stopCh
	}()
	return nil
}

// enqueueByPod enqueues the jointInferenceService object of the specified pod.
func (jc *JointInferenceServiceController) enqueueByPod(pod *v1.Pod, immediate bool) {
	controllerRef := metav1.GetControllerOf(pod)

	if controllerRef == nil {
		return
	}

	if controllerRef.Kind != jointServiceControllerKind.Kind {
		return
	}

	service, err := jc.serviceLister.JointInferenceServices(pod.Namespace).Get(controllerRef.Name)
	if err != nil {
		return
	}

	if service.UID != controllerRef.UID {
		return
	}

	jc.enqueueController(service, immediate)
}

// When a pod is created, enqueue the controller that manages it and update it's expectations.
func (jc *JointInferenceServiceController) addPod(obj interface{}) {
	pod := obj.(*v1.Pod)
	if pod.DeletionTimestamp != nil {
		// on a restart of the controller, it's possible a new pod shows up in a state that
		// is already pending deletion. Prevent the pod from being a creation observation.
		jc.deletePod(pod)
		return
	}

	// backoff to queue when PodFailed
	immediate := pod.Status.Phase != v1.PodFailed

	jc.enqueueByPod(pod, immediate)
}

// When a pod is updated, figure out what joint inference service manage it and wake them up.
func (jc *JointInferenceServiceController) updatePod(old, cur interface{}) {
	curPod := cur.(*v1.Pod)
	oldPod := old.(*v1.Pod)

	// no pod update, no queue
	if curPod.ResourceVersion == oldPod.ResourceVersion {
		return
	}

	jc.addPod(curPod)
}

// deletePod enqueues the jointinferenceservice obj When a pod is deleted
func (jc *JointInferenceServiceController) deletePod(obj interface{}) {
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
	jc.enqueueByPod(pod, true)
}

// obj could be an *sednav1.JointInferenceService, or a DeletionFinalStateUnknown marker item,
// immediate tells the controller to update the status right away, and should
// happen ONLY when there was a successful pod run.
func (jc *JointInferenceServiceController) enqueueController(obj interface{}, immediate bool) {
	key, err := k8scontroller.KeyFunc(obj)
	if err != nil {
		klog.Warningf("Couldn't get key for object %+v: %v", obj, err)
		return
	}

	backoff := time.Duration(0)
	if !immediate {
		backoff = getBackoff(jc.queue, key)
	}
	jc.queue.AddAfter(key, backoff)
}

// worker runs a worker thread that just dequeues items, processes them, and marks them done.
// It enforces that the sync is never invoked concurrently with the same key.
func (jc *JointInferenceServiceController) worker() {
	for jc.processNextWorkItem() {
	}
}

func (jc *JointInferenceServiceController) processNextWorkItem() bool {
	key, quit := jc.queue.Get()
	if quit {
		return false
	}
	defer jc.queue.Done(key)

	forget, err := jc.sync(key.(string))
	if err == nil {
		if forget {
			jc.queue.Forget(key)
		}
		return true
	}

	klog.Warningf("Error syncing jointinference service: %v", err)
	jc.queue.AddRateLimited(key)

	return true
}

// sync will sync the jointinferenceservice with the given key.
// This function is not meant to be invoked concurrently with the same key.
func (jc *JointInferenceServiceController) sync(key string) (bool, error) {
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
	sharedJointinferenceservice, err := jc.serviceLister.JointInferenceServices(ns).Get(name)
	if err != nil {
		if errors.IsNotFound(err) {
			klog.V(4).Infof("JointInferenceService has been deleted: %v", key)
			return true, nil
		}
		return false, err
	}

	jointinferenceservice := *sharedJointinferenceservice

	// if jointinferenceservice was finished previously, we don't want to redo the termination
	if isJointinferenceserviceFinished(&jointinferenceservice) {
		return true, nil
	}

	// set kind for jointinferenceservice in case that the kind is None
	// more details at https://github.com/kubernetes/kubernetes/issues/3030
	jointinferenceservice.SetGroupVersionKind(jointServiceControllerKind)

	selector, _ := GenerateSelector(&jointinferenceservice)
	pods, err := jc.podStore.Pods(jointinferenceservice.Namespace).List(selector)

	if err != nil {
		return false, err
	}

	klog.V(4).Infof("list jointinference service %v/%v, %v pods: %v", jointinferenceservice.Namespace, jointinferenceservice.Name, len(pods), pods)

	latestConditionLen := len(jointinferenceservice.Status.Conditions)

	active := calcActivePodCount(pods)
	var failed int32 = 0
	// neededCounts means that two pods should be created successfully in a jointinference service currently
	// two pods consist of edge pod and cloud pod
	var neededCounts int32 = 2
	// jointinferenceservice first start
	if jointinferenceservice.Status.StartTime == nil {
		now := metav1.Now()
		jointinferenceservice.Status.StartTime = &now
	} else {
		failed = neededCounts - active
	}

	var manageServiceErr error
	serviceFailed := false

	var latestConditionType sednav1.JointInferenceServiceConditionType = ""

	// get the latest condition type
	// based on that condition updated is appended, not inserted.
	jobConditions := jointinferenceservice.Status.Conditions
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
		message = "the worker of Jointinferenceservice failed"
		newCondtionType = sednav1.JointInferenceServiceCondFailed
		jc.recorder.Event(&jointinferenceservice, v1.EventTypeWarning, reason, message)
	} else {
		if len(pods) == 0 {
			active, manageServiceErr = jc.createPod(&jointinferenceservice)
		}
		if manageServiceErr != nil {
			serviceFailed = true
			message = error.Error(manageServiceErr)
			newCondtionType = sednav1.JointInferenceServiceCondFailed
			failed = neededCounts - active
		} else {
			// TODO: handle the case that the pod phase is PodSucceeded
			newCondtionType = sednav1.JointInferenceServiceCondRunning
		}
	}

	//
	if newCondtionType != latestConditionType {
		jointinferenceservice.Status.Conditions = append(jointinferenceservice.Status.Conditions, NewJointInferenceServiceCondition(newCondtionType, reason, message))
	}
	forget := false

	// no need to update the jointinferenceservice if the status hasn't changed since last time
	if jointinferenceservice.Status.Active != active || jointinferenceservice.Status.Failed != failed || len(jointinferenceservice.Status.Conditions) != latestConditionLen {
		jointinferenceservice.Status.Active = active
		jointinferenceservice.Status.Failed = failed

		if err := jc.updateStatus(&jointinferenceservice); err != nil {
			return forget, err
		}

		if serviceFailed && !isJointinferenceserviceFinished(&jointinferenceservice) {
			// returning an error will re-enqueue jointinferenceservice after the backoff period
			return forget, fmt.Errorf("failed pod(s) detected for jointinference service key %q", key)
		}

		forget = true
	}

	return forget, manageServiceErr
}

// NewJointInferenceServiceCondition creates a new joint condition
func NewJointInferenceServiceCondition(conditionType sednav1.JointInferenceServiceConditionType, reason, message string) sednav1.JointInferenceServiceCondition {
	return sednav1.JointInferenceServiceCondition{
		Type:               conditionType,
		Status:             v1.ConditionTrue,
		LastHeartbeatTime:  metav1.Now(),
		LastTransitionTime: metav1.Now(),
		Reason:             reason,
		Message:            message,
	}
}

func (jc *JointInferenceServiceController) updateStatus(jointinferenceservice *sednav1.JointInferenceService) error {
	serviceClient := jc.client.JointInferenceServices(jointinferenceservice.Namespace)
	var err error
	for i := 0; i <= statusUpdateRetries; i = i + 1 {
		var newJointinferenceservice *sednav1.JointInferenceService
		newJointinferenceservice, err = serviceClient.Get(context.TODO(), jointinferenceservice.Name, metav1.GetOptions{})
		if err != nil {
			break
		}
		newJointinferenceservice.Status = jointinferenceservice.Status
		if _, err = serviceClient.UpdateStatus(context.TODO(), newJointinferenceservice, metav1.UpdateOptions{}); err == nil {
			break
		}
	}
	return nil
}

func isJointinferenceserviceFinished(j *sednav1.JointInferenceService) bool {
	for _, c := range j.Status.Conditions {
		if (c.Type == sednav1.JointInferenceServiceCondFailed) && c.Status == v1.ConditionTrue {
			return true
		}
	}
	return false
}

func (jc *JointInferenceServiceController) createPod(service *sednav1.JointInferenceService) (active int32, err error) {
	active = 0

	// create pod for cloudPod
	err = jc.createCloudPod(service)
	if err != nil {
		return active, err
	}
	active++

	// create kubernetesService for cloudPod, and get bigServicePort for edgePod
	var bigServicePort int32
	bigModelIP, err := GetNodeIPByName(jc.kubeClient, service.Spec.CloudWorker.NodeName)
	bigServicePort, err = CreateKubernetesService(jc.kubeClient, service, bigModelPort, bigModelIP)
	if err != nil {
		return active, err
	}

	// create pod for edgePod
	err = jc.createEdgePod(service, bigServicePort)
	if err != nil {
		return active, err
	}
	active++

	return active, err
}

func (jc *JointInferenceServiceController) createCloudPod(service *sednav1.JointInferenceService) error {
	// deliver pod for cloudworker
	ctx := context.Background()
	var cloudModelPath string
	cloudModelName := service.Spec.CloudWorker.Model.Name
	cloudModel, err := jc.client.Models(service.Namespace).Get(ctx, cloudModelName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get cloud model %s: %w",
			cloudModelName, err)
	}

	cloudModelPath = cloudModel.Spec.URL

	// convert crd to json, and put them into env of container
	cloudModelJSON, _ := json.Marshal(cloudModel)
	cloudModelString := string(cloudModelJSON)
	cloudModelParent := filepath.Dir(cloudModelPath)

	cloudWorker := service.Spec.CloudWorker
	cloudCodePath := cloudWorker.WorkerSpec.ScriptDir
	cloudParameterJSON, _ := json.Marshal(cloudWorker.WorkerSpec.Parameters)
	cloudParameterString := string(cloudParameterJSON)

	// Container VolumeMounts parameters
	cloudCodeConPath := codePrefix
	cloudModelConPath := dataPrefix + cloudModelParent

	// Env parameters for cloud
	cloudModelURL := dataPrefix + cloudModelPath

	// Configure container mounting and Env information by initial ContainerPara
	var cloudContainer *ContainerPara = new(ContainerPara)
	cloudContainer.volumeMountList = []string{cloudCodeConPath, cloudModelConPath}
	cloudContainer.volumeList = []string{cloudCodePath, cloudModelParent}
	cloudContainer.volumeMapName = []string{"code", "model"}
	cloudContainer.env = map[string]string{
		"MODEL":               cloudModelString,
		"WORKER_NAME":         "cloudworker-" + utilrand.String(5),
		"SERVICE_NAME":        service.Name,
		"PARAMETERS":          cloudParameterString,
		"MODEL_URL":           cloudModelURL,
		"NAMESPACE":           service.Namespace,
		"BIG_MODEL_BIND_PORT": strconv.Itoa(int(bigModelPort)),
	}

	// create cloud pod
	err = jc.generatedPod(service, jointInferenceForCloud, cloudContainer, false)
	if err != nil {
		return err
	}
	return nil
}

func (jc *JointInferenceServiceController) createEdgePod(service *sednav1.JointInferenceService, bigServicePort int32) error {
	// deliver pod for edgeworker
	ctx := context.Background()
	edgeModelName := service.Spec.EdgeWorker.Model.Name
	edgeModel, err := jc.client.Models(service.Namespace).Get(ctx, edgeModelName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get edge model %s: %w",
			edgeModelName, err)
	}
	edgeModelPath := edgeModel.Spec.URL

	// get bigModelIP from nodeName in cloudWorker
	bigModelIP, err := GetNodeIPByName(jc.kubeClient, service.Spec.CloudWorker.NodeName)
	if err != nil {
		return fmt.Errorf("failed to get node ip: %w", err)
	}

	// convert crd to json, and put them into env of container
	edgeModelJSON, _ := json.Marshal(edgeModel)
	edgeModelString := string(edgeModelJSON)
	edgeModelParent := filepath.Dir(edgeModelPath)

	edgeWorker := service.Spec.EdgeWorker
	edgeCodePath := edgeWorker.WorkerSpec.ScriptDir
	edgeParameterJSON, _ := json.Marshal(edgeWorker.WorkerSpec.Parameters)
	edgeParameterString := string(edgeParameterJSON)
	HEMParameterJSON, _ := json.Marshal(edgeWorker.HardExampleMining.Parameters)
	HEMParameterString := string(HEMParameterJSON)

	// Container VolumeMounts parameters
	edgeCodeConPath := codePrefix
	edgeModelConPath := dataPrefix + edgeModelParent

	// Env parameters for edge
	edgeModelURL := dataPrefix + edgeModelPath

	// Configure container mounting and Env information by initial ContainerPara
	var edgeContainer *ContainerPara = new(ContainerPara)
	edgeContainer.volumeMountList = []string{edgeCodeConPath, edgeModelConPath}
	edgeContainer.volumeList = []string{edgeCodePath, edgeModelParent}
	edgeContainer.volumeMapName = []string{"code", "model"}
	edgeContainer.env = map[string]string{
		"MODEL":          edgeModelString,
		"WORKER_NAME":    "edgeworker-" + utilrand.String(5),
		"SERVICE_NAME":   service.Name,
		"BIG_MODEL_IP":   bigModelIP,
		"BIG_MODEL_PORT": strconv.Itoa(int(bigServicePort)),
		"PARAMETERS":     edgeParameterString,
		"HEM_PARAMETERS": HEMParameterString,
		"MODEL_URL":      edgeModelURL,
		"NAMESPACE":      service.Namespace,
		"HEM_NAME":       edgeWorker.HardExampleMining.Name,
		"LC_SERVER":      jc.cfg.LC.Server,
	}

	// create edge pod
	err = jc.generatedPod(service, jointInferenceForEdge, edgeContainer, true)
	if err != nil {
		return err
	}
	return nil
}

func (jc *JointInferenceServiceController) generatedPod(service *sednav1.JointInferenceService, podtype jointInferenceType,
	containerPara *ContainerPara, hostNetwork bool) error {
	var workerSpec sednav1.CommonWorkerSpec
	var volumeMounts []v1.VolumeMount
	var volumes []v1.Volume
	var envs []v1.EnvVar
	var nodeName string
	ctx := context.Background()
	if podtype == jointInferenceForEdge {
		workerSpec = service.Spec.EdgeWorker.WorkerSpec
		nodeName = service.Spec.EdgeWorker.NodeName
	} else {
		workerSpec = service.Spec.CloudWorker.WorkerSpec
		nodeName = service.Spec.CloudWorker.NodeName
	}
	// get baseImgURL from imageHub based on user's configuration in job CRD
	frameName := workerSpec.FrameworkType
	frameVersion := workerSpec.FrameworkVersion
	baseImgURL, err := MatchContainerBaseImage(jc.cfg.ImageHub, frameName, frameVersion)
	// TODO: if matched image is empty, the pod creation process will not proceed, return error directly.
	if err != nil {
		klog.Warningf("jointinference service %v/%v %v worker matching container base image occurs error:%v", service.Namespace, service.Name, podtype, err)
		return fmt.Errorf("%s pod occurs error: %w",
			podtype, err)
	}
	volumeMounts, volumes = CreateVolumeMap(containerPara)
	envs = CreateEnvVars(containerPara.env)
	podSpec := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace:    service.Namespace,
			GenerateName: service.Name + "-" + strings.ToLower(string(podtype)) + "-",
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(service, jointServiceControllerKind),
			},
			Labels: GenerateLabels(service),
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			NodeName:      nodeName,
			Containers: []v1.Container{
				{Name: "container-" + service.Name + "-" + strings.ToLower(string(podtype)) + "-" + utilrand.String(5),
					Image:        baseImgURL,
					Args:         []string{workerSpec.ScriptBootFile},
					Env:          envs,
					VolumeMounts: volumeMounts,
				}},
			Volumes:     volumes,
			HostNetwork: hostNetwork,
		},
	}
	pod, err := jc.kubeClient.CoreV1().Pods(service.Namespace).Create(ctx, podSpec, metav1.CreateOptions{})
	if err != nil {
		klog.Warningf("failed to create %s pod %s for jointinference service %v/%v, err:%s", string(podtype), pod.Name, service.Namespace, service.Name, err)
		return err
	}
	klog.V(2).Infof("%s pod %s is created successfully for jointinference service %v/%v", string(podtype), pod.Name, service.Namespace, service.Name)
	return nil
}

// GetName returns the name of the joint inference controller
func (jc *JointInferenceServiceController) GetName() string {
	return "JointInferenceServiceController"
}

// NewJointController creates a new JointInferenceService controller that keeps the relevant pods
// in sync with their corresponding JointInferenceService objects.
func NewJointController(cfg *config.ControllerConfig) (FeatureControllerI, error) {
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
	serviceInformer := serviceInformerFactory.Sedna().V1alpha1().JointInferenceServices()

	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: kubeClient.CoreV1().Events("")})

	jc := &JointInferenceServiceController{
		kubeClient: kubeClient,
		client:     crdclient.SednaV1alpha1(),

		queue:    workqueue.NewNamedRateLimitingQueue(workqueue.NewItemExponentialFailureRateLimiter(DefaultBackOff, MaxBackOff), "jointinferenceservice"),
		recorder: eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "jointinferenceservice-controller"}),
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

	stopCh := messageContext.Done()
	kubeInformerFactory.Start(stopCh)
	serviceInformerFactory.Start(stopCh)
	return jc, err
}
