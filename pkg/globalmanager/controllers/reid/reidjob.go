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

package reid

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
	KindName = "ReidJob"
	// Name is this controller name
	Name = "Reid"
	// ReidWorker is this name given to the worker pod
	ReidWorker = "reid"
	// ReidPort is the port where the service will be exposed
	ReidPort = 5000
)

// Kind contains the schema.GroupVersionKind for this controller type.
var Kind = sednav1.SchemeGroupVersion.WithKind(KindName)

// Controller ensures that all ReidJob objects have corresponding pods to
// run their configured workload.
type Controller struct {
	kubeClient kubernetes.Interface
	client     sednaclientset.SednaV1alpha1Interface

	// podStoreSynced returns true if the pod store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	podStoreSynced cache.InformerSynced
	// jobStoreSynced returns true if the ReidJob store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	jobStoreSynced cache.InformerSynced

	// A store of jobs
	jobLister sednav1listers.ReidJobLister

	// A store of pods, populated by the podController
	podStore corelisters.PodLister

	// FLJobs that need to be updated
	queue workqueue.RateLimitingInterface

	recorder record.EventRecorder

	cfg *config.ControllerConfig

	sendToEdgeFunc runtime.DownstreamSendFunc
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

// enqueueByPod enqueues the ReidJob object of the specified pod.
func (c *Controller) enqueueByPod(pod *v1.Pod, immediate bool) {
	controllerRef := metav1.GetControllerOf(pod)

	if controllerRef == nil {
		return
	}

	if controllerRef.Kind != Kind.Kind {
		return
	}

	job, err := c.jobLister.ReidJobs(pod.Namespace).Get(controllerRef.Name)
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

// When a pod is updated, figure out what Reid job manage it and wake them up.
func (c *Controller) updatePod(old, cur interface{}) {
	curPod := cur.(*v1.Pod)
	oldPod := old.(*v1.Pod)

	// no pod update, no queue
	if curPod.ResourceVersion == oldPod.ResourceVersion {
		return
	}

	c.addPod(curPod)
}

// deletePod enqueues the ReidJob obj When a pod is deleted
func (c *Controller) deletePod(obj interface{}) {
	pod, ok := obj.(*v1.Pod)

	// comment from https://github.com/kubernetes/kubernetes/blob/master/pkg/controller/job/job_controller.go

	// When a delete is dropped, the relist will notice a pod in the store not
	// in the list, leading to the insertion of a tombstone object which contains
	// the deleted key/value. Note that this value might be stale. If the pod
	// changed labels the new ReidJob will not be woken up till the periodic resync.
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

// obj could be an *sednav1.ReidJob, or a DeletionFinalStateUnknown marker item,
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

	klog.Warningf("Error syncing Reid job: %v", err)
	c.queue.AddRateLimited(key)

	return true
}

// sync will sync the ReidJob with the given key if it has had its expectations fulfilled, meaning
// it did not expect to see any more of its pods created or deleted. This function is not meant to be invoked
// concurrently with the same key.
func (c *Controller) sync(key string) (bool, error) {
	startTime := time.Now()
	defer func() {
		klog.V(4).Infof("Finished syncing Reid job %q (%v)", key, time.Since(startTime))
	}()

	ns, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return false, err
	}
	if len(ns) == 0 || len(name) == 0 {
		return false, fmt.Errorf("invalid Reid job key %q: either namespace or name is missing", key)
	}
	sharedJob, err := c.jobLister.ReidJobs(ns).Get(name)
	if err != nil {
		if errors.IsNotFound(err) {
			klog.V(4).Infof("%s %v has been deleted", Name, key)
			return true, nil
		}
		return false, err
	}

	job := *sharedJob
	// set kind for ReidJob in case that the kind is None
	job.SetGroupVersionKind(Kind)

	// if job was finished previously, we don't want to redo the termination
	if IsJobFinished(&job) {
		return true, nil
	}

	selector, _ := runtime.GenerateSelector(&job)
	pods, err := c.podStore.Pods(job.Namespace).List(selector)
	if err != nil {
		return false, err
	}

	activePods := k8scontroller.FilterActivePods(pods)
	active := int32(len(activePods))
	succeeded, failed := countPods(pods)
	conditions := len(job.Status.Conditions)

	// set StartTime when job is handled firstly
	if job.Status.StartTime == nil {
		now := metav1.Now()
		job.Status.StartTime = &now
	}

	var manageJobErr error
	jobFailed := false
	var failureReason string
	var failureMessage string
	phase := job.Status.Phase

	if failed > 0 {
		jobFailed = true
		failureReason = "workerFailed"
		failureMessage = "the worker of ReidJob failed"
	}

	if jobFailed {
		job.Status.Conditions = append(job.Status.Conditions, NewJobCondition(sednav1.ReidJobCondFailed, failureReason, failureMessage))
		job.Status.Phase = sednav1.ReidJobFailed
		c.recorder.Event(&job, v1.EventTypeWarning, failureReason, failureMessage)
	} else {
		// in the First time, we create the pods
		if len(pods) == 0 {
			active, manageJobErr = c.createJob(&job)
		}
		complete := false
		if succeeded > 0 && active == 0 {
			complete = true
		}
		if complete {
			job.Status.Conditions = append(job.Status.Conditions, NewJobCondition(sednav1.ReidJobCondCompleted, "", ""))
			now := metav1.Now()
			job.Status.CompletionTime = &now
			c.recorder.Event(&job, v1.EventTypeNormal, "Completed", "ReidJob completed")
			job.Status.Phase = sednav1.ReidJobSucceeded
		} else {
			job.Status.Phase = sednav1.ReidJobRunning
		}
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
			// returning an error will re-enqueue ReidJob after the backoff period
			return forget, fmt.Errorf("failed pod(s) detected for ReidJob key %q", key)
		}

		forget = true
	}

	return forget, manageJobErr
}

func NewJobCondition(conditionType sednav1.ReidJobConditionType, reason, message string) sednav1.ReidJobCondition {
	return sednav1.ReidJobCondition{
		Type:               conditionType,
		Status:             v1.ConditionTrue,
		LastHeartbeatTime:  metav1.Now(),
		LastTransitionTime: metav1.Now(),
		Reason:             reason,
		Message:            message,
	}
}

// countPods returns number of succeeded and failed pods
func countPods(pods []*v1.Pod) (succeeded, failed int32) {
	succeeded = int32(filterPods(pods, v1.PodSucceeded))
	failed = int32(filterPods(pods, v1.PodFailed))
	return
}

func (c *Controller) updateJobStatus(job *sednav1.ReidJob) error {
	jobClient := c.client.ReidJobs(job.Namespace)
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

func IsJobFinished(j *sednav1.ReidJob) bool {
	for _, c := range j.Status.Conditions {
		if (c.Type == sednav1.ReidJobCondCompleted || c.Type == sednav1.ReidJobCondFailed) && c.Status == v1.ConditionTrue {
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

func (c *Controller) createJob(job *sednav1.ReidJob) (active int32, err error) {
	active = 0

	// Configure training worker's mounts and envs
	var workerParam runtime.WorkerParam

	workerParam.Env = map[string]string{
		"WORKER_NAME": utilrand.String(5),
		"JOB_NAME":    job.Name,
		"NAMESPACE":   job.Namespace,
		"LC_SERVER":   c.cfg.LC.Server,
	}
	workerParam.WorkerType = "reid"
	workerParam.RestartPolicy = job.Spec.Template.Spec.RestartPolicy

	if job.Spec.KafkaSupport {
		workerParam.Env["KAFKA_ENABLED"] = strconv.FormatBool(job.Spec.KafkaSupport)
	}

	// create reid worker pod based on configured parameters
	_, err = runtime.CreatePodWithTemplate(c.kubeClient, job, &job.Spec.Template, &workerParam)
	if err != nil {
		return active, fmt.Errorf("failed to create reid worker: %w", err)
	}

	active++

	return active, nil
}

// New creates a new reid job controller that keeps the relevant pods
// in sync with their corresponding ReidJob objects.
func New(cc *runtime.ControllerContext) (runtime.FeatureControllerI, error) {
	cfg := cc.Config

	podInformer := cc.KubeInformerFactory.Core().V1().Pods()

	jobInformer := cc.SednaInformerFactory.Sedna().V1alpha1().ReidJobs()

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

			// when a video analytics job is added,
			// send it to edge's LC.
			fc.syncToEdge(watch.Added, obj)
		},
		UpdateFunc: func(old, cur interface{}) {
			fc.enqueueController(cur, true)

			// when a video analytics job is updated,
			// send it to edge's LC as Added event.
			fc.syncToEdge(watch.Added, cur)
		},
		DeleteFunc: func(obj interface{}) {
			fc.enqueueController(obj, true)

			// when a video analytics job is deleted,
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
