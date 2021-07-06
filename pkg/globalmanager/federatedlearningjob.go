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
	"strconv"
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

const (
	FLJobStageAgg   = "Aggregation"
	FLJobStageTrain = "Training"
)

// flJobControllerKind contains the schema.GroupVersionKind for this controller type.
var flJobControllerKind = sednav1.SchemeGroupVersion.WithKind("FederatedLearningJob")

// FederatedController ensures that all FLJob objects have corresponding pods to
// run their configured workload.
type FederatedController struct {
	kubeClient kubernetes.Interface
	client     sednaclientset.SednaV1alpha1Interface

	// podStoreSynced returns true if the pod store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	podStoreSynced cache.InformerSynced
	// jobStoreSynced returns true if the flJob store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	jobStoreSynced cache.InformerSynced

	// A store of jobs
	jobLister sednav1listers.FederatedLearningJobLister

	// A store of pods, populated by the podController
	podStore corelisters.PodLister

	// FLJobs that need to be updated
	queue workqueue.RateLimitingInterface

	recorder record.EventRecorder

	cfg *config.ControllerConfig
}

// Run the main goroutine responsible for watching and syncing jobs.
func (fc *FederatedController) Start() error {
	workers := 1
	stopCh := messageContext.Done()

	go func() {
		defer utilruntime.HandleCrash()
		defer fc.queue.ShutDown()
		klog.Infof("Starting federatedlearning job controller")
		defer klog.Infof("Shutting down federatedlearning job controller")

		if !cache.WaitForNamedCacheSync("federatedlearning job", stopCh, fc.podStoreSynced, fc.jobStoreSynced) {
			klog.Errorf("failed to wait for caches to sync")

			return
		}

		klog.Infof("Starting federatedlearning job workers")
		for i := 0; i < workers; i++ {
			go wait.Until(fc.worker, time.Second, stopCh)
		}

		<-stopCh
	}()
	return nil
}

// enqueueByPod enqueues the FederatedLearningJob object of the specified pod.
func (fc *FederatedController) enqueueByPod(pod *v1.Pod, immediate bool) {
	controllerRef := metav1.GetControllerOf(pod)

	if controllerRef == nil {
		return
	}

	if controllerRef.Kind != flJobControllerKind.Kind {
		return
	}

	job, err := fc.jobLister.FederatedLearningJobs(pod.Namespace).Get(controllerRef.Name)
	if err != nil {
		return
	}

	if job.UID != controllerRef.UID {
		return
	}

	fc.enqueueController(job, immediate)
}

// When a pod is created, enqueue the controller that manages it and update it's expectations.
func (fc *FederatedController) addPod(obj interface{}) {
	pod := obj.(*v1.Pod)
	if pod.DeletionTimestamp != nil {
		// on a restart of the controller, it's possible a new pod shows up in a state that
		// is already pending deletion. Prevent the pod from being a creation observation.
		fc.deletePod(pod)
		return
	}

	// backoff to queue when PodFailed
	immediate := pod.Status.Phase != v1.PodFailed

	fc.enqueueByPod(pod, immediate)
}

// When a pod is updated, figure out what federatedlearning job manage it and wake them up.
func (fc *FederatedController) updatePod(old, cur interface{}) {
	curPod := cur.(*v1.Pod)
	oldPod := old.(*v1.Pod)

	// no pod update, no queue
	if curPod.ResourceVersion == oldPod.ResourceVersion {
		return
	}

	fc.addPod(curPod)
}

// deletePod enqueues the FederatedLearningJob obj When a pod is deleted
func (fc *FederatedController) deletePod(obj interface{}) {
	pod, ok := obj.(*v1.Pod)

	// comment from https://github.com/kubernetes/kubernetes/blob/master/pkg/controller/job/job_controller.go

	// When a delete is dropped, the relist will notice a pod in the store not
	// in the list, leading to the insertion of a tombstone object which contains
	// the deleted key/value. Note that this value might be stale. If the pod
	// changed labels the new FederatedLearningJob will not be woken up till the periodic resync.
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
	fc.enqueueByPod(pod, true)
}

// obj could be an *sednav1.FederatedLearningJob, or a DeletionFinalStateUnknown marker item,
// immediate tells the controller to update the status right away, and should
// happen ONLY when there was a successful pod run.
func (fc *FederatedController) enqueueController(obj interface{}, immediate bool) {
	key, err := k8scontroller.KeyFunc(obj)
	if err != nil {
		klog.Warningf("Couldn't get key for object %+v: %v", obj, err)
		return
	}

	backoff := time.Duration(0)
	if !immediate {
		backoff = getBackoff(fc.queue, key)
	}
	fc.queue.AddAfter(key, backoff)
}

// worker runs a worker thread that just dequeues items, processes them, and marks them done.
// It enforces that the syncHandler is never invoked concurrently with the same key.
func (fc *FederatedController) worker() {
	for fc.processNextWorkItem() {
	}
}

func (fc *FederatedController) processNextWorkItem() bool {
	key, quit := fc.queue.Get()
	if quit {
		return false
	}
	defer fc.queue.Done(key)

	forget, err := fc.syncFLJob(key.(string))
	if err == nil {
		if forget {
			fc.queue.Forget(key)
		}
		return true
	}

	klog.Warningf("Error syncing federatedlearning job: %v", err)
	fc.queue.AddRateLimited(key)

	return true
}

// syncFLJob will sync the flJob with the given key if it has had its expectations fulfilled, meaning
// it did not expect to see any more of its pods created or deleted. This function is not meant to be invoked
// concurrently with the same key.
func (fc *FederatedController) syncFLJob(key string) (bool, error) {
	startTime := time.Now()
	defer func() {
		klog.V(4).Infof("Finished syncing federatedlearning job %q (%v)", key, time.Since(startTime))
	}()

	ns, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return false, err
	}
	if len(ns) == 0 || len(name) == 0 {
		return false, fmt.Errorf("invalid federatedlearning job key %q: either namespace or name is missing", key)
	}
	sharedFLJob, err := fc.jobLister.FederatedLearningJobs(ns).Get(name)
	if err != nil {
		if errors.IsNotFound(err) {
			klog.V(4).Infof("FLJob has been deleted: %v", key)
			return true, nil
		}
		return false, err
	}
	flJob := *sharedFLJob
	// set kind for flJob in case that the kind is None
	flJob.SetGroupVersionKind(sednav1.SchemeGroupVersion.WithKind("FederatedLearningJob"))
	// if flJob was finished previously, we don't want to redo the termination
	if IsFLJobFinished(&flJob) {
		return true, nil
	}
	selector, _ := GenerateSelector(&flJob)
	pods, err := fc.podStore.Pods(flJob.Namespace).List(selector)
	if err != nil {
		return false, err
	}

	activePods := k8scontroller.FilterActivePods(pods)
	active := int32(len(activePods))
	succeeded, failed := getStatus(pods)
	conditions := len(flJob.Status.Conditions)
	// flJob first start
	if flJob.Status.StartTime == nil {
		now := metav1.Now()
		flJob.Status.StartTime = &now
	}

	var manageJobErr error
	jobFailed := false
	var failureReason string
	var failureMessage string
	phase := flJob.Status.Phase

	if failed > 0 {
		jobFailed = true
		failureReason = "workerFailed"
		failureMessage = "the worker of FLJob failed"
	}

	if jobFailed {
		flJob.Status.Conditions = append(flJob.Status.Conditions, NewFLJobCondition(sednav1.FLJobCondFailed, failureReason, failureMessage))
		flJob.Status.Phase = sednav1.FLJobFailed
		fc.recorder.Event(&flJob, v1.EventTypeWarning, failureReason, failureMessage)
	} else {
		// in the First time, we create the pods
		if len(pods) == 0 {
			active, manageJobErr = fc.createPod(&flJob)
		}
		complete := false
		if succeeded > 0 && active == 0 {
			complete = true
		}
		if complete {
			flJob.Status.Conditions = append(flJob.Status.Conditions, NewFLJobCondition(sednav1.FLJobCondComplete, "", ""))
			now := metav1.Now()
			flJob.Status.CompletionTime = &now
			fc.recorder.Event(&flJob, v1.EventTypeNormal, "Completed", "FLJob completed")
			flJob.Status.Phase = sednav1.FLJobSucceeded
		} else {
			flJob.Status.Phase = sednav1.FLJobRunning
		}
	}

	forget := false
	// Check if the number of jobs succeeded increased since the last check. If yes "forget" should be true
	// This logic is linked to the issue: https://github.com/kubernetes/kubernetes/issues/56853 that aims to
	// improve the FLJob backoff policy when parallelism > 1 and few FLJobs failed but others succeed.
	// In this case, we should clear the backoff delay.
	if flJob.Status.Succeeded < succeeded {
		forget = true
	}

	// no need to update the flJob if the status hasn't changed since last time
	if flJob.Status.Active != active || flJob.Status.Succeeded != succeeded || flJob.Status.Failed != failed || len(flJob.Status.Conditions) != conditions || flJob.Status.Phase != phase {
		flJob.Status.Active = active
		flJob.Status.Succeeded = succeeded
		flJob.Status.Failed = failed

		if jobFailed && !IsFLJobFinished(&flJob) {
			// returning an error will re-enqueue FLJob after the backoff period
			return forget, fmt.Errorf("failed pod(s) detected for flJob key %q", key)
		}

		forget = true
	}

	return forget, manageJobErr
}

func NewFLJobCondition(conditionType sednav1.FLJobConditionType, reason, message string) sednav1.FLJobCondition {
	return sednav1.FLJobCondition{
		Type:              conditionType,
		Status:            v1.ConditionTrue,
		LastProbeTime:     metav1.Now(),
		LastHeartbeatTime: metav1.Now(),
		Reason:            reason,
		Message:           message,
	}
}

// getStatus returns no of succeeded and failed pods running a flJob
func getStatus(pods []*v1.Pod) (succeeded, failed int32) {
	succeeded = int32(filterPods(pods, v1.PodSucceeded))
	failed = int32(filterPods(pods, v1.PodFailed))
	return
}

func (fc *FederatedController) updateFLJobStatus(flJob *sednav1.FederatedLearningJob) error {
	jobClient := fc.client.FederatedLearningJobs(flJob.Namespace)
	var err error
	for i := 0; i <= ResourceUpdateRetries; i = i + 1 {
		var newFLJob *sednav1.FederatedLearningJob
		newFLJob, err = jobClient.Get(context.TODO(), flJob.Name, metav1.GetOptions{})
		if err != nil {
			break
		}
		newFLJob.Status = flJob.Status
		if _, err = jobClient.UpdateStatus(context.TODO(), newFLJob, metav1.UpdateOptions{}); err == nil {
			break
		}
	}
	return nil
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

func IsFLJobFinished(j *sednav1.FederatedLearningJob) bool {
	for _, c := range j.Status.Conditions {
		if (c.Type == sednav1.FLJobCondComplete || c.Type == sednav1.FLJobCondFailed) && c.Status == v1.ConditionTrue {
			return true
		}
	}
	return false
}

func (fc *FederatedController) createPod(job *sednav1.FederatedLearningJob) (active int32, err error) {
	active = 0
	ctx := context.Background()

	modelName := job.Spec.AggregationWorker.Model.Name
	model, err := fc.client.Models(job.Namespace).Get(ctx, modelName, metav1.GetOptions{})
	if err != nil {
		return active, fmt.Errorf("failed to get model %s: %w",
			modelName, err)
	}

	secretName := model.Spec.CredentialName
	var modelSecret *v1.Secret
	if secretName != "" {
		modelSecret, _ = fc.kubeClient.CoreV1().Secrets(job.Namespace).Get(context.TODO(), secretName, metav1.GetOptions{})
	}

	participantsCount := strconv.Itoa(len(job.Spec.TrainingWorkers))

	// deliver pod for aggregation worker
	aggWorker := job.Spec.AggregationWorker

	// Configure container mounting and Env information by initial WorkerParam
	var aggPort int32 = 7363
	var aggWorkerParam *WorkerParam = new(WorkerParam)
	aggWorkerParam.env = map[string]string{
		"NAMESPACE":   job.Namespace,
		"WORKER_NAME": "aggworker-" + utilrand.String(5),
		"JOB_NAME":    job.Name,

		"AGG_BIND_PORT":      strconv.Itoa(int(aggPort)),
		"PARTICIPANTS_COUNT": participantsCount,
	}

	aggWorkerParam.workerType = FLJobStageAgg
	aggWorkerParam.restartPolicy = v1.RestartPolicyOnFailure

	aggWorkerParam.mounts = append(aggWorkerParam.mounts,
		WorkerMount{
			URL: &MountURL{
				URL:                   model.Spec.URL,
				Secret:                modelSecret,
				DownloadByInitializer: false,
			},
			EnvName: "MODEL_URL",
		},
	)

	// create aggpod based on configured parameters
	_, err = createPodWithTemplate(fc.kubeClient, job, &aggWorker.Template, aggWorkerParam)
	if err != nil {
		return active, err
	}
	active++

	var appIP string
	var aggServicePort int32

	// FIXME(llhuii): only the case that Spec.NodeName specified is support,
	// will support Spec.NodeSelector.
	appIP, err = GetNodeIPByName(fc.kubeClient, job.Spec.AggregationWorker.Template.Spec.NodeName)

	aggServicePort, err = CreateKubernetesService(fc.kubeClient, job, FLJobStageAgg, aggPort, appIP)
	if err != nil {
		return active, err
	}
	// deliver pod for training worker
	for _, trainingWorker := range job.Spec.TrainingWorkers {
		// get dataseturl through parsing crd of dataset
		datasetName := trainingWorker.Dataset.Name
		dataset, err := fc.client.Datasets(job.Namespace).Get(ctx, datasetName, metav1.GetOptions{})
		if err != nil {
			return active, fmt.Errorf("failed to get dataset %s: %w",
				datasetName, err)
		}

		secretName := dataset.Spec.CredentialName
		var datasetSecret *v1.Secret
		if secretName != "" {
			datasetSecret, _ = fc.kubeClient.CoreV1().Secrets(job.Namespace).Get(context.TODO(), secretName, metav1.GetOptions{})
		}

		// Configure container mounting and Env information by initial WorkerParam
		var workerParam *WorkerParam = new(WorkerParam)

		workerParam.mounts = append(workerParam.mounts,
			WorkerMount{
				URL: &MountURL{
					URL:    model.Spec.URL,
					Secret: modelSecret,
				},
				EnvName: "MODEL_URL",
			},

			WorkerMount{
				URL: &MountURL{
					URL:    dataset.Spec.URL,
					Secret: datasetSecret,
				},
				EnvName: "TRAIN_DATASET_URL",
			},
		)

		workerParam.env = map[string]string{
			"AGG_PORT": strconv.Itoa(int(aggServicePort)),
			"AGG_IP":   appIP,

			"WORKER_NAME":        "trainworker-" + utilrand.String(5),
			"JOB_NAME":           job.Name,
			"PARTICIPANTS_COUNT": participantsCount,
			"NAMESPACE":          job.Namespace,
			"MODEL_NAME":         modelName,
			"DATASET_NAME":       datasetName,
			"LC_SERVER":          fc.cfg.LC.Server,
		}
		workerParam.workerType = TrainPodType
		workerParam.hostNetwork = true
		workerParam.restartPolicy = v1.RestartPolicyOnFailure
		// create train pod based on configured parameters
		_, err = createPodWithTemplate(fc.kubeClient, job, &trainingWorker.Template, workerParam)
		if err != nil {
			return active, err
		}
		active++
	}
	return
}

func (fc *FederatedController) GetName() string {
	return "FederatedLearningJobController"
}

// NewFederatedController creates a new FederatedLearningJob controller that keeps the relevant pods
// in sync with their corresponding FFederatedLearningJob objects.
func NewFederatedController(cfg *config.ControllerConfig) (FeatureControllerI, error) {
	namespace := cfg.Namespace
	if namespace == "" {
		namespace = metav1.NamespaceAll
	}
	kubeClient, err := utils.KubeClient()
	kubecfg, _ := utils.KubeConfig()
	crdclient, err := clientset.NewForConfig(kubecfg)
	kubeInformerFactory := kubeinformers.NewSharedInformerFactoryWithOptions(kubeClient, time.Second*30, kubeinformers.WithNamespace(namespace))

	podInformer := kubeInformerFactory.Core().V1().Pods()

	jobInformerFactory := informers.NewSharedInformerFactoryWithOptions(crdclient, time.Second*30, informers.WithNamespace(namespace))
	jobInformer := jobInformerFactory.Sedna().V1alpha1().FederatedLearningJobs()

	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: kubeClient.CoreV1().Events("")})

	fc := &FederatedController{
		kubeClient: kubeClient,
		client:     crdclient.SednaV1alpha1(),

		queue:    workqueue.NewNamedRateLimitingQueue(workqueue.NewItemExponentialFailureRateLimiter(DefaultBackOff, MaxBackOff), "flJob"),
		recorder: eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "flJob-controller"}),
		cfg:      cfg,
	}

	jobInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			fc.enqueueController(obj, true)
		},
		UpdateFunc: func(old, cur interface{}) {
			fc.enqueueController(cur, true)
		},
		DeleteFunc: func(obj interface{}) {
			fc.enqueueController(obj, true)
		},
	})
	fc.jobLister = jobInformer.Lister()
	fc.jobStoreSynced = jobInformer.Informer().HasSynced

	podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    fc.addPod,
		UpdateFunc: fc.updatePod,
		DeleteFunc: fc.deletePod,
	})
	fc.podStore = podInformer.Lister()
	fc.podStoreSynced = podInformer.Informer().HasSynced

	stopCh := make(chan struct{})
	kubeInformerFactory.Start(stopCh)
	jobInformerFactory.Start(stopCh)
	return fc, err
}
