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

package federatedlearning

import (
	"context"
	"fmt"
	"strconv"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	utilrand "k8s.io/apimachinery/pkg/util/rand"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
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
	sednaclientset "github.com/kubeedge/sedna/pkg/client/clientset/versioned/typed/sedna/v1alpha1"
	sednav1listers "github.com/kubeedge/sedna/pkg/client/listers/sedna/v1alpha1"
	"github.com/kubeedge/sedna/pkg/globalmanager/config"
	"github.com/kubeedge/sedna/pkg/globalmanager/runtime"
)

const (
	// KindName is the kind name of CR this controller controls
	KindName = "FederatedLearningJob"
	// Name is this controller name
	Name = "FederatedLearning"
)

const (
	jobStageAgg   = "Aggregation"
	jobStageTrain = "Training"
)

// Kind contains the schema.GroupVersionKind for this controller type.
var Kind = sednav1.SchemeGroupVersion.WithKind(KindName)

// Controller ensures that all FederatedLearningJob objects have corresponding pods to
// run their configured workload.
type Controller struct {
	kubeClient kubernetes.Interface
	client     sednaclientset.SednaV1alpha1Interface

	// podStoreSynced returns true if the pod store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	podStoreSynced cache.InformerSynced
	// jobStoreSynced returns true if the FederatedLearningJob store has been synced at least once.
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

	sendToEdgeFunc runtime.DownstreamSendFunc

	// map to record the pods that are recreated
	recreatedPods sync.Map

	flSelector labels.Selector

	aggServiceHost string

	preventRecreation bool
}

// Run starts the main goroutine responsible for watching and syncing jobs.
func (c *Controller) Run(stopCh <-chan struct{}) {
	workers := 1

	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()

	klog.Infof("Starting %s controller", Name)
	defer klog.Infof("Shutting down %s controller", Name)

	if !cache.WaitForNamedCacheSync(Name, stopCh, c.podStoreSynced, c.jobStoreSynced) {
		klog.Errorf("failed to wait for %s caches to sync", Name)

		return
	}

	klog.Infof("Starting %s workers", Name)
	for i := 0; i < workers; i++ {
		go wait.Until(c.worker, time.Second, stopCh)
	}

	<-stopCh
}

// enqueueByPod enqueues the FederatedLearningJob object of the specified pod.
func (c *Controller) enqueueByPod(pod *v1.Pod, immediate bool) {
	controllerRef := metav1.GetControllerOf(pod)

	if controllerRef == nil {
		return
	}

	if controllerRef.Kind != Kind.Kind {
		return
	}

	job, err := c.jobLister.FederatedLearningJobs(pod.Namespace).Get(controllerRef.Name)
	if err != nil {
		return
	}

	if job.UID != controllerRef.UID {
		return
	}

	c.enqueueController(job, immediate)
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

// When a pod is updated, figure out what federatedlearning job manage it and wake them up.
func (c *Controller) updatePod(old, cur interface{}) {
	curPod := cur.(*v1.Pod)
	oldPod := old.(*v1.Pod)

	// no pod update, no queue
	if curPod.ResourceVersion == oldPod.ResourceVersion {
		return
	}

	c.addPod(curPod)
}

// deletePod enqueues the FederatedLearningJob obj When a pod is deleted
func (c *Controller) deletePod(obj interface{}) {
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
	c.enqueueByPod(pod, true)

	// when the CRD is updated, do not recreate the pod
	// if c.preventRecreation is true, do not recreate the pod
	if c.preventRecreation {
		return
	}
	// if pod is manually deleted, recreate it
	// first check if the pod is owned by a FederatedLearningJob
	controllerRef := metav1.GetControllerOf(pod)
	if controllerRef == nil || controllerRef.Kind != Kind.Kind {
		return
	}

	// then check if the pod is already in the map
	if _, exists := c.recreatedPods.Load(pod.Name); exists {
		return
	}

	// if not, recreate it
	klog.Infof("Pod %s/%s deleted, recreating...", pod.Namespace, pod.Name)
	// Create a deep copy of the old pod
	newPod := pod.DeepCopy()
	// Reset the resource version and UID as they are unique to each object
	newPod.ResourceVersion = ""
	newPod.UID = ""
	// Clear the status
	newPod.Status = v1.PodStatus{}
	// Remove the deletion timestamp
	newPod.DeletionTimestamp = nil
	// Remove the deletion grace period seconds
	newPod.DeletionGracePeriodSeconds = nil
	_, err := c.kubeClient.CoreV1().Pods(pod.Namespace).Create(context.TODO(), newPod, metav1.CreateOptions{})
	if err != nil {
		return
	}
	klog.Infof("Successfully recreated pod %s/%s", newPod.Namespace, newPod.Name)
	// mark the pod as recreated
	c.recreatedPods.Store(newPod.Name, true)
	// set a timer to delete the record from the map after a while
	go func() {
		time.Sleep(5 * time.Second)
		c.recreatedPods.Delete(pod.Name)
	}()
}

// obj could be an *sednav1.FederatedLearningJob, or a DeletionFinalStateUnknown marker item,
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
// It enforces that the syncHandler is never invoked concurrently with the same key.
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

	klog.Warningf("Error syncing federatedlearning job: %v", err)
	c.queue.AddRateLimited(key)

	return true
}

// sync will sync the FederatedLearningJob with the given key if it has had its expectations fulfilled, meaning
// it did not expect to see any more of its pods created or deleted. This function is not meant to be invoked
// concurrently with the same key.
func (c *Controller) sync(key string) (bool, error) {
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
	sharedJob, err := c.jobLister.FederatedLearningJobs(ns).Get(name)
	if err != nil {
		if errors.IsNotFound(err) {
			klog.V(4).Infof("%s %v has been deleted", Name, key)
			return true, nil
		}
		return false, err
	}

	job := *sharedJob
	// set kind for FederatedLearningJob in case that the kind is None
	job.SetGroupVersionKind(Kind)

	// if job was finished previously, we don't want to redo the termination
	if IsJobFinished(&job) {
		return true, nil
	}

	c.flSelector, _ = runtime.GenerateSelector(&job)
	pods, err := c.podStore.Pods(job.Namespace).List(c.flSelector)
	if err != nil {
		return false, err
	}

	activePods := k8scontroller.FilterActivePods(pods)
	active := int32(len(activePods))
	var activeAgg int32
	var activeTrain int32
	succeeded, failed := countPods(pods)
	conditions := len(job.Status.Conditions)

	// set StartTime when job is handled firstly
	if job.Status.StartTime == nil {
		now := metav1.Now()
		job.Status.StartTime = &now
	}

	var manageJobErr error
	var manageAggErr error
	var manageTrainErr error
	jobFailed := false
	var failureReason string
	var failureMessage string
	phase := job.Status.Phase

	if failed > 0 {
		jobFailed = true
		failureReason = "workerFailed"
		failureMessage = "the worker of FederatedLearningJob failed"
	}

	if jobFailed {
		job.Status.Conditions = append(job.Status.Conditions, NewJobCondition(sednav1.FLJobCondFailed, failureReason, failureMessage))
		job.Status.Phase = sednav1.FLJobFailed
		c.recorder.Event(&job, v1.EventTypeWarning, failureReason, failureMessage)
	} else {
		// in the First time, we create the pods
		if len(pods) == 0 {
			activeAgg, manageAggErr = c.createAggPod(&job)
			createServiceErr := c.createService(&job)
			if createServiceErr != nil {
				return false, createServiceErr
			}
			activeTrain, manageTrainErr = c.createTrainPod(&job)
			active = activeAgg + activeTrain
		}
		complete := false
		if succeeded > 0 && active == 0 {
			complete = true
		}
		if complete {
			job.Status.Conditions = append(job.Status.Conditions, NewJobCondition(sednav1.FLJobCondComplete, "", ""))
			now := metav1.Now()
			job.Status.CompletionTime = &now
			c.recorder.Event(&job, v1.EventTypeNormal, "Completed", "FederatedLearningJob completed")
			job.Status.Phase = sednav1.FLJobSucceeded
		} else {
			job.Status.Phase = sednav1.FLJobRunning
		}
	}

	// Combine manageAggErr and manageTrainErr into a single error
	if manageAggErr != nil || manageTrainErr != nil {
		manageJobErr = fmt.Errorf("aggregator error: %v, training error: %v", manageAggErr, manageTrainErr)
	}
	forget := false
	// Check if the number of jobs succeeded increased since the last check. If yes "forget" should be true
	// This logic is linked to the issue: https://github.com/kubernetes/kubernetes/issues/56853 that aims to
	// improve the job backoff policy when parallelism > 1 and few FLJobs failed but others succeed.
	// In this case, we should clear the backoff delay.
	if job.Status.Succeeded < succeeded {
		forget = true
	}

	// no need to update the job if the status hasn't changed since last time
	if job.Status.Active != active || job.Status.Succeeded != succeeded || job.Status.Failed != failed || len(job.Status.Conditions) != conditions || job.Status.Phase != phase {
		job.Status.Active = active
		job.Status.Succeeded = succeeded
		job.Status.Failed = failed
		c.updateJobStatus(&job)

		if jobFailed && !IsJobFinished(&job) {
			// returning an error will re-enqueue FederatedLearningJob after the backoff period
			return forget, fmt.Errorf("failed pod(s) detected for FederatedLearningJob key %q", key)
		}

		forget = true
	}

	return forget, manageJobErr
}

func NewJobCondition(conditionType sednav1.FLJobConditionType, reason, message string) sednav1.FLJobCondition {
	return sednav1.FLJobCondition{
		Type:              conditionType,
		Status:            v1.ConditionTrue,
		LastProbeTime:     metav1.Now(),
		LastHeartbeatTime: metav1.Now(),
		Reason:            reason,
		Message:           message,
	}
}

// countPods returns number of succeeded and failed pods
func countPods(pods []*v1.Pod) (succeeded, failed int32) {
	succeeded = int32(filterPods(pods, v1.PodSucceeded))
	failed = int32(filterPods(pods, v1.PodFailed))
	return
}

func (c *Controller) updateJobStatus(job *sednav1.FederatedLearningJob) error {
	jobClient := c.client.FederatedLearningJobs(job.Namespace)
	return runtime.RetryUpdateStatus(job.Name, job.Namespace, func() error {
		newJob, err := jobClient.Get(context.TODO(), job.Name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		newJob.Status = job.Status
		_, err = jobClient.UpdateStatus(context.TODO(), newJob, metav1.UpdateOptions{})
		return err
	})
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

func IsJobFinished(j *sednav1.FederatedLearningJob) bool {
	for _, c := range j.Status.Conditions {
		if (c.Type == sednav1.FLJobCondComplete || c.Type == sednav1.FLJobCondFailed) && c.Status == v1.ConditionTrue {
			return true
		}
	}
	return false
}

func (c *Controller) getSecret(namespace, name, ownerStr string) (secret *v1.Secret, err error) {
	if name != "" {
		secret, err = c.kubeClient.CoreV1().Secrets(namespace).Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			err = fmt.Errorf("failed to get the secret %s for %s: %w",
				name,
				ownerStr, err)
		}
	}

	return
}

func (c *Controller) getModelAndItsSecret(ctx context.Context, namespace, name string) (model *sednav1.Model, secret *v1.Secret, err error) {
	if name != "" {
		model, err = c.client.Models(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			err = fmt.Errorf("failed to get the model %s: %w", name, err)
		}
	}

	if model != nil {
		secret, err = c.getSecret(
			namespace,
			model.Spec.CredentialName,
			fmt.Sprintf("model %s", name),
		)
	}

	return
}

func (c *Controller) getDatasetAndItsSecret(ctx context.Context, namespace, name string) (dataset *sednav1.Dataset, secret *v1.Secret, err error) {
	if name != "" {
		dataset, err = c.client.Datasets(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			err = fmt.Errorf("failed to get the dataset %s: %w", name, err)
		}
	}

	if dataset != nil {
		secret, err = c.getSecret(
			namespace,
			dataset.Spec.CredentialName,
			fmt.Sprintf("model %s", name),
		)
	}

	return
}

// addWorkerMount adds CR(e.g., model, dataset)'s url to worker mount.
func (c *Controller) addWorkerMount(workerParam *runtime.WorkerParam, url string, envName string,
	secret *v1.Secret, downloadByInitializer bool) {
	if url != "" {
		workerParam.Mounts = append(workerParam.Mounts,
			runtime.WorkerMount{
				URL: &runtime.MountURL{
					URL:                   url,
					Secret:                secret,
					DownloadByInitializer: downloadByInitializer,
				},
				EnvName: envName,
			},
		)
	}
}

// addTransmitterToWorkerParam adds transmitter to the WorkerParam
func (c *Controller) addTransmitterToWorkerParam(param *runtime.WorkerParam, job *sednav1.FederatedLearningJob) error {
	transmitter := job.Spec.Transmitter

	if transmitter.S3 != nil {
		param.Env["TRANSMITTER"] = "s3"
		url := transmitter.S3.AggregationDataPath
		secret, err := c.getSecret(
			job.Namespace,
			transmitter.S3.CredentialName,
			fmt.Sprintf("for aggregationData: %s", url))
		if err != nil {
			return err
		}
		param.Mounts = append(param.Mounts,
			runtime.WorkerMount{
				URL: &runtime.MountURL{
					URL:    url,
					Secret: secret,
				},
				EnvName: "AGG_DATA_PATH",
			},
		)
	} else {
		param.Env["TRANSMITTER"] = "ws"
	}

	return nil
}
func (c *Controller) createAggPod(job *sednav1.FederatedLearningJob) (active int32, err error) {
	active = 0
	ctx := context.Background()

	pretrainedModelName := job.Spec.PretrainedModel.Name
	pretrainedModel, pretrainedModelSecret, err := c.getModelAndItsSecret(ctx, job.Namespace, pretrainedModelName)
	if err != nil {
		return active, err
	}

	modelName := job.Spec.AggregationWorker.Model.Name
	model, modelSecret, err := c.getModelAndItsSecret(ctx, job.Namespace, modelName)
	if err != nil {
		return active, fmt.Errorf("failed to get aggregation model: %w", err)
	}

	participantsCount := strconv.Itoa(len(job.Spec.TrainingWorkers))

	// deliver pod for aggregation worker
	aggWorker := job.Spec.AggregationWorker

	// Configure aggregation worker's mounts and envs
	var aggPort int32 = 7363
	var aggWorkerParam runtime.WorkerParam

	aggWorkerParam.Env = map[string]string{
		"NAMESPACE":   job.Namespace,
		"WORKER_NAME": "aggworker-" + utilrand.String(5),
		"JOB_NAME":    job.Name,

		"AGG_BIND_PORT":      strconv.Itoa(int(aggPort)),
		"PARTICIPANTS_COUNT": participantsCount,
	}

	if err := c.addTransmitterToWorkerParam(&aggWorkerParam, job); err != nil {
		return active, fmt.Errorf("failed to add transmitter to worker param: %w", err)
	}

	aggWorkerParam.WorkerType = jobStageAgg
	aggWorkerParam.RestartPolicy = v1.RestartPolicyOnFailure

	c.addWorkerMount(&aggWorkerParam, model.Spec.URL, "MODEL_URL",
		modelSecret, true)

	if pretrainedModel != nil {
		c.addWorkerMount(&aggWorkerParam, pretrainedModel.Spec.URL, "PRETRAINED_MODEL_URL",
			pretrainedModelSecret, true)
	}
	aggWorker.Template.Name = fmt.Sprintf("%s-aggworker", job.Name)
	// create aggpod based on configured parameters
	_, err = runtime.CreatePodWithTemplate(c.kubeClient, job, &aggWorker.Template, &aggWorkerParam)
	if err != nil {
		return active, fmt.Errorf("failed to create aggregation worker: %w", err)
	}
	klog.Infof("create aggpod success")
	active++
	return
}

func (c *Controller) createTrainPod(job *sednav1.FederatedLearningJob) (active int32, err error) {
	active = 0
	ctx := context.Background()

	pretrainedModelName := job.Spec.PretrainedModel.Name
	pretrainedModel, pretrainedModelSecret, err := c.getModelAndItsSecret(ctx, job.Namespace, pretrainedModelName)
	if err != nil {
		return active, fmt.Errorf("failed to get pretrained model: %w", err)
	}

	modelName := job.Spec.AggregationWorker.Model.Name
	model, modelSecret, err := c.getModelAndItsSecret(ctx, job.Namespace, modelName)
	if err != nil {
		return active, fmt.Errorf("failed to get aggregation model: %w", err)
	}

	var aggPort int32 = 7363
	participantsCount := strconv.Itoa(len(job.Spec.TrainingWorkers))

	// deliver pod for training worker
	for i, trainingWorker := range job.Spec.TrainingWorkers {
		// Configure training worker's mounts and envs
		var workerParam runtime.WorkerParam

		c.addWorkerMount(&workerParam, model.Spec.URL, "MODEL_URL", modelSecret, true)

		if pretrainedModel != nil {
			c.addWorkerMount(&workerParam, pretrainedModel.Spec.URL, "PRETRAINED_MODEL_URL",
				pretrainedModelSecret, true)
		}

		datasetName := trainingWorker.Dataset.Name
		dataset, datasetSecret, err := c.getDatasetAndItsSecret(ctx, job.Namespace, datasetName)
		if err != nil {
			return active, err
		}

		c.addWorkerMount(&workerParam, dataset.Spec.URL, "TRAIN_DATASET_URL",
			datasetSecret, true)

		workerParam.Env = map[string]string{
			"AGG_PORT": strconv.Itoa(int(aggPort)),
			"AGG_IP":   c.aggServiceHost,

			"WORKER_NAME":        "trainworker-" + utilrand.String(5),
			"JOB_NAME":           job.Name,
			"PARTICIPANTS_COUNT": participantsCount,
			"NAMESPACE":          job.Namespace,
			"MODEL_NAME":         modelName,
			"DATASET_NAME":       datasetName,
			"LC_SERVER":          c.cfg.LC.Server,
		}

		workerParam.WorkerType = runtime.TrainPodType
		workerParam.HostNetwork = true
		workerParam.RestartPolicy = v1.RestartPolicyOnFailure

		if err := c.addTransmitterToWorkerParam(&workerParam, job); err != nil {
			return active, fmt.Errorf("failed to add transmitter to worker param: %w", err)
		}
		trainingWorker.Template.Name = fmt.Sprintf("%s-trainworker-%d", job.Name, i)
		// create training worker based on configured parameters
		_, err = runtime.CreatePodWithTemplate(c.kubeClient, job, &trainingWorker.Template, &workerParam)
		if err != nil {
			return active, fmt.Errorf("failed to create %dth training worker: %w", i, err)
		}
		active++
	}
	return
}

// New creates a new federated learning job controller that keeps the relevant pods
// in sync with their corresponding FederatedLearningJob objects.
func New(cc *runtime.ControllerContext) (runtime.FeatureControllerI, error) {
	cfg := cc.Config

	podInformer := cc.KubeInformerFactory.Core().V1().Pods()

	jobInformer := cc.SednaInformerFactory.Sedna().V1alpha1().FederatedLearningJobs()

	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: cc.KubeClient.CoreV1().Events("")})

	fc := &Controller{
		kubeClient: cc.KubeClient,
		client:     cc.SednaClient.SednaV1alpha1(),

		queue:    workqueue.NewNamedRateLimitingQueue(workqueue.NewItemExponentialFailureRateLimiter(runtime.DefaultBackOff, runtime.MaxBackOff), Name),
		recorder: eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: Name + "-controller"}),
		cfg:      cfg,
	}

	jobInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			fc.enqueueController(obj, true)

			// when a federated learning job is added,
			// send it to edge's LC.
			fc.syncToEdge(watch.Added, obj)
		},
		UpdateFunc: fc.updateJob,

		DeleteFunc: func(obj interface{}) {
			fc.enqueueController(obj, true)

			// when a federated learning job is deleted,
			// send it to edge's LC.
			fc.syncToEdge(watch.Deleted, obj)
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

	return fc, nil
}

func (c *Controller) updateJob(old, cur interface{}) {
	oldJob, ok := old.(*sednav1.FederatedLearningJob)
	if !ok {
		return
	}
	curJob, ok := cur.(*sednav1.FederatedLearningJob)
	if !ok {
		return
	}

	if oldJob.ResourceVersion == curJob.ResourceVersion {
		return
	}

	if oldJob.Generation != curJob.Generation {
		pods, err := c.podStore.Pods(curJob.Namespace).List(c.flSelector)
		if err != nil {
			klog.Errorf("Failed to list pods: %v", err)
		}
		c.preventRecreation = true
		for _, pod := range pods {
			// delete all pods
			c.kubeClient.CoreV1().Pods(pod.Namespace).Delete(context.TODO(), pod.Name, metav1.DeleteOptions{})
			klog.Infof("CRD modified, so we deleted pod %s/%s", pod.Namespace, pod.Name)
		}
		klog.Infof("CRD modified, so we deleted all pods, and will create new pods")
		curJob.SetGroupVersionKind(Kind)
		_, err = c.createAggPod(curJob)
		if err != nil {
			klog.Errorf("Failed to create aggregation worker: %v", err)
		}
		_, err = c.createTrainPod(curJob)
		if err != nil {
			klog.Errorf("Failed to create training workers: %v", err)
		}
		// update the job status
		c.client.FederatedLearningJobs(curJob.Namespace).Update(context.TODO(), curJob, metav1.UpdateOptions{})
	}

	c.preventRecreation = false
	c.enqueueController(curJob, true)

	// when a federated learning job is updated,
	// send it to edge's LC as Added event.
	c.syncToEdge(watch.Added, curJob)
}

// create edgemesh service for the job
func (c *Controller) createService(job *sednav1.FederatedLearningJob) (err error) {
	var aggPort int32 = 7363
	c.aggServiceHost, err = runtime.CreateEdgeMeshService(c.kubeClient, job, jobStageAgg, aggPort)
	if err != nil {
		return err
	}
	return nil
}
