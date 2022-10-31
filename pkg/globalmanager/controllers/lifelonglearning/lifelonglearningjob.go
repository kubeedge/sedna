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

package lifelonglearning

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilrand "k8s.io/apimachinery/pkg/util/rand"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes"
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
	KindName = "LifelongLearningJob"
	// Name is this controller name
	Name = "LifelongLearning"
)

// Kind contains the schema.GroupVersionKind for this controller type.
var Kind = sednav1.SchemeGroupVersion.WithKind(KindName)

// Controller ensures that all LifelongLearningJob objects have corresponding pods to
// run their configured workload.
type Controller struct {
	kubeClient kubernetes.Interface
	client     sednaclientset.SednaV1alpha1Interface

	// podStoreSynced returns true if the pod store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	podStoreSynced cache.InformerSynced
	// jobStoreSynced returns true if the lifelonglearningjob store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	jobStoreSynced cache.InformerSynced

	// A store of jobs
	jobLister sednav1listers.LifelongLearningJobLister

	// A store of pods, populated by the podController
	podStore corelisters.PodLister

	// LifelongLearningJobs that need to be updated
	queue workqueue.RateLimitingInterface

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

// enqueueByPod enqueues the lifelonglearningjob object of the specified pod.
func (c *Controller) enqueueByPod(pod *v1.Pod, immediate bool) {
	controllerRef := metav1.GetControllerOf(pod)

	if controllerRef == nil {
		return
	}

	if controllerRef.Kind != Kind.Kind {
		return
	}

	service, err := c.jobLister.LifelongLearningJobs(pod.Namespace).Get(controllerRef.Name)
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

// When a pod is updated, figure out what lifelonglearning job manage it and wake them up.
func (c *Controller) updatePod(old, cur interface{}) {
	curPod := cur.(*v1.Pod)
	oldPod := old.(*v1.Pod)

	// no pod update, no queue
	if curPod.ResourceVersion == oldPod.ResourceVersion {
		return
	}

	c.addPod(curPod)
}

// deletePod enqueues the lifelonglearningjob obj When a pod is deleted
func (c *Controller) deletePod(obj interface{}) {
	pod, ok := obj.(*v1.Pod)

	// comment from https://github.com/kubernetes/kubernetes/blob/master/pkg/controller/job/job_controller.go

	// When a delete is dropped, the relist will notice a pod in the store not
	// in the list, leading to the insertion of a tombstone object which contains
	// the deleted key/value. Note that this value might be stale. If the pod
	// changed labels the new lifelonglearningjob will not be woken up till the periodic resync.
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

// obj could be an *sedna.LifelongLearningJob, or a DeletionFinalStateUnknown marker item,
// immediate tells the controller to update the status right away, and should
// happen ONLY when there was a successful pod run.
func (c *Controller) enqueueController(obj interface{}, immediate bool) {
	key, err := k8scontroller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for object %+v: %v", obj, err))
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

	utilruntime.HandleError(fmt.Errorf("Error syncing lifelonglearning job: %v", err))
	c.queue.AddRateLimited(key)

	return true
}

// sync will sync the lifelonglearning job with the given key if it has had its expectations fulfilled, meaning
// it did not expect to see any more of its pods created or deleted. This function is not meant to be invoked
// concurrently with the same key.
func (c *Controller) sync(key string) (bool, error) {
	startTime := time.Now()
	defer func() {
		klog.V(4).Infof("Finished syncing lifelonglearning job %q (%v)", key, time.Since(startTime))
	}()

	ns, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return false, err
	}
	if len(ns) == 0 || len(name) == 0 {
		return false, fmt.Errorf("invalid lifelonglearning job key %q: either namespace or name is missing", key)
	}
	sharedJob, err := c.jobLister.LifelongLearningJobs(ns).Get(name)
	if err != nil {
		if errors.IsNotFound(err) {
			klog.V(4).Infof("lifelonglearning job has been deleted: %v", key)
			return true, nil
		}
		return false, err
	}
	job := *sharedJob
	// set kind for lifelonglearningjob in case that the kind is None
	job.SetGroupVersionKind(Kind)

	if job.Status.StartTime == nil {
		// job is first in
		now := metav1.Now()
		job.Status.StartTime = &now
	}

	// if job was finished previously, we don't want to redo the termination
	if IsJobFinished(&job) {
		return true, nil
	}

	forget := false
	jobFailed := false
	needUpdated := false

	// transit this job's state machine
	needUpdated, err = c.transitJobState(&job)
	if err != nil {
		klog.V(2).Infof("lifelonglearning job %v/%v failed to be updated, err:%s", job.Namespace, job.Name, err)
	}

	if needUpdated {
		if err := c.updateJobStatus(&job); err != nil {
			return forget, err
		}

		if jobFailed && !IsJobFinished(&job) {
			// returning an error will re-enqueue LifelongLearningJob after the backoff period
			return forget, fmt.Errorf("failed pod(s) detected for lifelonglearningjob key %q", key)
		}

		forget = true
	}

	return forget, err
}

// setWorkerNodeNameOfJob sets the worker nodeName of the specified job
// which is used for downstream to sync job info to the specified LC located in nodeName.
func (c *Controller) setWorkerNodeNameOfJob(job *sednav1.LifelongLearningJob, jobStage string, nodeName string) error {
	key := runtime.AnnotationsKeyPrefix + jobStage

	return c.addJobAnnotations(job, key, nodeName)
}

// addJobAnnotations adds info in job annotations
func (c *Controller) addJobAnnotations(job *sednav1.LifelongLearningJob, key string, value string) error {
	ann := job.GetAnnotations()
	if ann[key] == value {
		// already set
		return nil
	}

	patchData := metav1.PartialObjectMetadata{
		ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{key: value}}}

	patchDataBytes, err := json.Marshal(&patchData)
	if err != nil {
		return err
	}

	jobClient := c.client.LifelongLearningJobs(job.Namespace)
	return runtime.RetryUpdateStatus(job.Name, job.Namespace, func() error {
		newJob, err := jobClient.Get(context.TODO(), job.Name, metav1.GetOptions{})
		if err != nil {
			return err
		}

		annotations := newJob.GetAnnotations()
		if annotations[key] == value {
			return nil
		}

		_, err = jobClient.Patch(context.TODO(), job.Name, types.MergePatchType, patchDataBytes, metav1.PatchOptions{})
		return err
	})
}

// transitJobState transit job to next state
func (c *Controller) transitJobState(job *sednav1.LifelongLearningJob) (bool, error) {
	var initialType sednav1.LLJobStageConditionType
	var latestCondition sednav1.LLJobCondition = sednav1.LLJobCondition{
		Stage: sednav1.LLJobTrain,
		Type:  initialType,
	}

	var newConditionType sednav1.LLJobStageConditionType
	var needUpdated = false

	var podStatus v1.PodPhase = v1.PodUnknown
	var pod *v1.Pod

	jobConditions := job.Status.Conditions
	if len(jobConditions) > 0 {
		// get latest pod and pod status
		latestCondition = (jobConditions)[len(jobConditions)-1]
		klog.V(2).Infof("lifelonglearning job %v/%v latest stage %v:", job.Namespace, job.Name,
			latestCondition.Stage)
		pod = c.getSpecifiedPods(job, string(latestCondition.Stage))

		if pod != nil {
			podStatus = pod.Status.Phase
		}
	}
	jobStage := latestCondition.Stage
	currentType := latestCondition.Type
	newConditionType = currentType

	switch currentType {
	case initialType:
		newConditionType = sednav1.LLJobStageCondWaiting

	case sednav1.LLJobStageCondWaiting:
		// do nothing, waiting for LC to set type from waiting to ready

	case sednav1.LLJobStageCondReady:
		// create a pod, and set type from ready to starting
		// include train, eval, deploy pod
		var err error
		if jobStage == sednav1.LLJobDeploy {
			err = c.restartInferPod(job)
			if err != nil {
				klog.V(2).Infof("lifelonglearning job %v/%v inference pod failed to restart, err:%s", job.Namespace, job.Name, err)
				return needUpdated, err
			}

			klog.V(2).Infof("lifelonglearning job %v/%v inference pod restarts successfully", job.Namespace, job.Name)
			newConditionType = sednav1.LLJobStageCondCompleted
		} else {
			if podStatus != v1.PodPending && podStatus != v1.PodRunning {
				err = c.createPod(job, jobStage)
				if err != nil {
					return needUpdated, err
				}
			}
			newConditionType = sednav1.LLJobStageCondStarting
		}

	case sednav1.LLJobStageCondStarting, sednav1.LLJobStageCondRunning:
		if podStatus == v1.PodRunning {
			// add nodeName to job
			if err := c.setWorkerNodeNameOfJob(job, string(jobStage), pod.Spec.NodeName); err != nil {
				return needUpdated, err
			}

			// watch pod status, if pod running, set type running
			newConditionType = sednav1.LLJobStageCondRunning
		} else if podStatus == v1.PodSucceeded {
			// watch pod status, if pod completed, set type completed
			newConditionType = sednav1.LLJobStageCondCompleted
			klog.V(2).Infof("lifelonglearning job %v/%v %v stage completed!", job.Namespace, job.Name, jobStage)
		} else if podStatus == v1.PodFailed {
			newConditionType = sednav1.LLJobStageCondFailed
			klog.V(2).Infof("lifelonglearning job %v/%v %v stage failed!", job.Namespace, job.Name, jobStage)
		}
	case sednav1.LLJobStageCondCompleted:
		jobStage = c.getNextStage(jobStage)
		newConditionType = sednav1.LLJobStageCondWaiting

	case sednav1.LLJobStageCondFailed:
		jobStage = sednav1.LLJobTrain
		newConditionType = sednav1.LLJobStageCondWaiting

	default:
		// do nothing when given other type out of cases
	}

	klog.V(2).Infof("lifelonglearning job %v/%v, conditions: %v", job.Namespace, job.Name, jobConditions)
	if latestCondition.Type != newConditionType {
		job.Status.Conditions = append(job.Status.Conditions, NewJobCondition(newConditionType, jobStage))
		needUpdated = true
		return needUpdated, nil
	}
	return needUpdated, nil
}

// updateJobStatus ensures that jobstatus can be updated rightly
func (c *Controller) updateJobStatus(job *sednav1.LifelongLearningJob) error {
	jobClient := c.client.LifelongLearningJobs(job.Namespace)
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

func NewJobCondition(conditionType sednav1.LLJobStageConditionType, jobStage sednav1.LLJobStage) sednav1.LLJobCondition {
	return sednav1.LLJobCondition{
		Type:               conditionType,
		Status:             v1.ConditionTrue,
		LastHeartbeatTime:  metav1.Now(),
		LastTransitionTime: metav1.Now(),
		Reason:             "",
		Message:            "",
		Stage:              jobStage,
	}
}

func (c *Controller) generatePodName(jobName string, workerType string) string {
	return jobName + "-" + strings.ToLower(workerType) + "-" + utilrand.String(5)
}

func (c *Controller) getSpecifiedPods(job *sednav1.LifelongLearningJob, podType string) *v1.Pod {
	if podType == "Deploy" {
		podType = runtime.InferencePodType
	}
	var latestPod *v1.Pod
	selector, _ := runtime.GenerateSelector(job)
	pods, err := c.podStore.Pods(job.Namespace).List(selector)
	if len(pods) == 0 || err != nil {
		return nil
	}
	var matchTag = false
	latestPod = pods[0]
	for _, pod := range pods {
		s := strings.Split(pod.Name, "-")
		CurrentPodType := s[len(s)-2]
		if (latestPod.CreationTimestamp.Before(&pod.CreationTimestamp) || latestPod.CreationTimestamp.Equal(&pod.CreationTimestamp)) && CurrentPodType == strings.ToLower(podType) {
			latestPod = pod
			matchTag = true
		}
	}
	if !matchTag {
		return nil
	}
	return latestPod
}

func (c *Controller) restartInferPod(job *sednav1.LifelongLearningJob) error {
	inferPod := c.getSpecifiedPods(job, runtime.InferencePodType)
	if inferPod == nil {
		klog.V(2).Infof("No inferpod is running in lifelonglearning job %v/%v", job.Namespace, job.Name)
		err := c.createInferPod(job)
		return err
	}
	ctx := context.Background()
	err := c.kubeClient.CoreV1().Pods(job.Namespace).Delete(ctx, inferPod.Name, metav1.DeleteOptions{})
	if err != nil {
		klog.Warningf("failed to delete inference pod %s for lifelonglearning job %v/%v, err:%s", inferPod.Name, job.Namespace, job.Name, err)
		return err
	}
	err = c.createInferPod(job)
	if err != nil {
		klog.Warningf("failed to create inference pod %s for lifelonglearning job %v/%v, err:%s", inferPod.Name, job.Namespace, job.Name, err)
		return err
	}
	return nil
}

func (c *Controller) getNextStage(currentStage sednav1.LLJobStage) sednav1.LLJobStage {
	switch currentStage {
	case sednav1.LLJobTrain:
		return sednav1.LLJobEval
	case sednav1.LLJobEval:
		return sednav1.LLJobDeploy
	case sednav1.LLJobDeploy:
		return sednav1.LLJobTrain
	default:
		return sednav1.LLJobTrain
	}
}

func (c *Controller) getSecret(namespace, name string, ownerStr string) (secret *v1.Secret, err error) {
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

func IsJobFinished(j *sednav1.LifelongLearningJob) bool {
	// TODO
	return false
}

// isCompletedInitialTraining checks whether job has completed initial train task.
func (c *Controller) hasCompletedInitialTraining(jobConditions []sednav1.LLJobCondition) bool {
	for i := 0; i < len(jobConditions); i++ {
		jobCond := jobConditions[i]
		if jobCond.Stage == sednav1.LLJobTrain && jobCond.Type == sednav1.LLJobStageCondCompleted {
			return true
		}
	}
	return false
}

func (c *Controller) getCloudKBIndex(jobConditions []sednav1.LLJobCondition) string {
	for i := len(jobConditions) - 1; i >= 0; i-- {
		jobCond := jobConditions[i]
		var cond ConditionData
		if jobCond.Stage == sednav1.LLJobTrain && jobCond.Type == sednav1.LLJobStageCondCompleted {
			if err := (&cond).Unmarshal([]byte(jobCond.Data)); err != nil {
				continue
			}

			if cond.Output == nil || len(cond.Output.Models) == 0 {
				continue
			}

			model := cond.Output.Models[0]
			return model.GetURL()
		}
	}
	return ""
}

func (c *Controller) createPod(job *sednav1.LifelongLearningJob, podtype sednav1.LLJobStage) (err error) {
	ctx := context.Background()
	var podTemplate *v1.PodTemplateSpec

	LLDatasetName := job.Spec.Dataset.Name

	dataset, err := c.client.Datasets(job.Namespace).Get(ctx, LLDatasetName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get dataset %s: %w", LLDatasetName, err)
	}

	datasetSecret, err := c.getSecret(
		job.Namespace,
		dataset.Spec.CredentialName,
		fmt.Sprintf("dataset %s", dataset.Name),
	)
	if err != nil {
		return err
	}

	jobSecret, err := c.getSecret(
		job.Namespace,
		job.Spec.CredentialName,
		fmt.Sprintf("lifelonglearning job %s", job.Name),
	)
	if err != nil {
		return err
	}

	jobConditions := job.Status.Conditions

	// get all url for train and eval from data in condition
	condDataStr := jobConditions[len(job.Status.Conditions)-1].Data
	klog.V(2).Infof("lifelonglearning job %v/%v data condition:%s", job.Namespace, job.Name, condDataStr)
	var cond ConditionData
	(&cond).Unmarshal([]byte(condDataStr))
	if cond.Input == nil {
		return fmt.Errorf("empty input from condData")
	}
	dataURL := cond.Input.DataURL
	inputmodelURLs := cond.GetInputModelURLs()

	var originalDataURLOrIndex string
	if cond.Input.DataIndexURL != "" {
		// this guarantee dataset.Spec.URL is not in host filesystem by LC,
		// but cond.Input.DataIndexURL could be in host filesystem.
		originalDataURLOrIndex = cond.Input.DataIndexURL
	} else {
		originalDataURLOrIndex = dataset.Spec.URL
	}

	var workerParam *runtime.WorkerParam = new(runtime.WorkerParam)
	if podtype == sednav1.LLJobTrain {
		workerParam.WorkerType = "Train"

		podTemplate = &job.Spec.TrainSpec.Template
		// Env parameters for train

		hasCompletedInitialTraining := c.hasCompletedInitialTraining(jobConditions)

		workerParam.Env = map[string]string{
			"NAMESPACE":                      job.Namespace,
			"JOB_NAME":                       job.Name,
			"WORKER_NAME":                    "train-worker-" + utilrand.String(5),
			"HAS_COMPLETED_INITIAL_TRAINING": strconv.FormatBool(hasCompletedInitialTraining),
			"LC_SERVER":                      c.cfg.LC.Server,
			"KB_SERVER":                      c.cfg.KB.Server,
		}

		if hasCompletedInitialTraining {
			workerParam.Env["CLOUD_KB_INDEX"] = c.getCloudKBIndex(jobConditions)
		}

		workerParam.Mounts = append(workerParam.Mounts,
			runtime.WorkerMount{
				URL: &runtime.MountURL{
					URL:                   cond.Input.OutputDir,
					Secret:                jobSecret,
					DownloadByInitializer: false,
				},
				EnvName: "OUTPUT_URL",
			},

			runtime.WorkerMount{
				URL: &runtime.MountURL{
					URL:                   dataURL,
					Secret:                jobSecret,
					DownloadByInitializer: true,
				},
				EnvName: "TRAIN_DATASET_URL",
			},

			// see https://github.com/kubeedge/sedna/issues/35
			runtime.WorkerMount{
				URL: &runtime.MountURL{
					Secret:                datasetSecret,
					URL:                   originalDataURLOrIndex,
					Indirect:              dataset.Spec.URL != originalDataURLOrIndex,
					DownloadByInitializer: true,
				},
				EnvName: "ORIGINAL_DATASET_URL",
			},
		)
	} else {
		podTemplate = &job.Spec.EvalSpec.Template
		workerParam.WorkerType = "Eval"

		// Configure Env information for eval by initial WorkerParam
		workerParam.Env = map[string]string{
			"NAMESPACE":   job.Namespace,
			"JOB_NAME":    job.Name,
			"WORKER_NAME": "eval-worker-" + utilrand.String(5),

			"LC_SERVER": c.cfg.LC.Server,
			"KB_SERVER": c.cfg.KB.Server,
		}

		var modelMountURLs []runtime.MountURL
		for _, url := range inputmodelURLs {
			modelMountURLs = append(modelMountURLs, runtime.MountURL{
				URL:                   url,
				Secret:                jobSecret,
				DownloadByInitializer: true,
			})
		}
		workerParam.Mounts = append(workerParam.Mounts,
			runtime.WorkerMount{
				URLs:    modelMountURLs,
				Name:    "models",
				EnvName: "MODEL_URLS",
			},

			runtime.WorkerMount{
				URL: &runtime.MountURL{
					URL:                   cond.Input.OutputDir,
					Secret:                jobSecret,
					DownloadByInitializer: false,
				},
				EnvName: "OUTPUT_URL",
			},

			runtime.WorkerMount{
				URL: &runtime.MountURL{
					URL:                   dataURL,
					Secret:                datasetSecret,
					DownloadByInitializer: true,
				},
				Name:    "datasets",
				EnvName: "TEST_DATASET_URL",
			},

			runtime.WorkerMount{
				URL: &runtime.MountURL{
					Secret:                datasetSecret,
					URL:                   originalDataURLOrIndex,
					DownloadByInitializer: true,
					Indirect:              dataset.Spec.URL != originalDataURLOrIndex,
				},
				Name:    "origin-dataset",
				EnvName: "ORIGINAL_DATASET_URL",
			},
		)
	}

	// set the default policy instead of Always policy
	workerParam.RestartPolicy = v1.RestartPolicyOnFailure
	workerParam.HostNetwork = true
	workerParam.DNSPolicy = v1.DNSClusterFirstWithHostNet

	// create pod based on podtype
	_, err = runtime.CreatePodWithTemplate(c.kubeClient, job, podTemplate, workerParam)
	if err != nil {
		return err
	}
	return
}

func (c *Controller) createInferPod(job *sednav1.LifelongLearningJob) error {
	inferModelURL := strings.Join([]string{strings.TrimRight(job.Spec.OutputDir, "/"), "deploy/index.pkl"}, "/")

	jobSecret, err := c.getSecret(
		job.Namespace,
		job.Spec.CredentialName,
		fmt.Sprintf("lifelonglearning job %s", job.Name),
	)
	if err != nil {
		return err
	}

	var workerParam *runtime.WorkerParam = new(runtime.WorkerParam)
	workerParam.Mounts = append(workerParam.Mounts,
		runtime.WorkerMount{
			URL: &runtime.MountURL{
				URL:                   inferModelURL,
				Secret:                jobSecret,
				DownloadByInitializer: false,
			},
			Name:    "models",
			EnvName: "MODEL_URLS",
		},
	)

	workerParam.Env = map[string]string{
		"NAMESPACE":   job.Namespace,
		"JOB_NAME":    job.Name,
		"WORKER_NAME": "inferworker-" + utilrand.String(5),

		"LC_SERVER": c.cfg.LC.Server,
	}

	workerParam.WorkerType = runtime.InferencePodType
	workerParam.HostNetwork = true
	workerParam.DNSPolicy = v1.DNSClusterFirstWithHostNet

	// create edge pod
	_, err = runtime.CreatePodWithTemplate(c.kubeClient, job, &job.Spec.DeploySpec.Template, workerParam)
	return err
}

// New creates a new LifelongLearningJob controller that keeps the relevant pods
// in sync with their corresponding LifelongLearningJob objects.
func New(cc *runtime.ControllerContext) (runtime.FeatureControllerI, error) {
	cfg := cc.Config

	podInformer := cc.KubeInformerFactory.Core().V1().Pods()

	jobInformer := cc.SednaInformerFactory.Sedna().V1alpha1().LifelongLearningJobs()

	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: cc.KubeClient.CoreV1().Events("")})

	jc := &Controller{
		kubeClient: cc.KubeClient,
		client:     cc.SednaClient.SednaV1alpha1(),
		queue:      workqueue.NewNamedRateLimitingQueue(workqueue.NewItemExponentialFailureRateLimiter(runtime.DefaultBackOff, runtime.MaxBackOff), Name),
		cfg:        cfg,
	}

	jobInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			jc.enqueueController(obj, true)
			jc.syncToEdge(watch.Added, obj)
		},
		UpdateFunc: func(old, cur interface{}) {
			jc.enqueueController(cur, true)
			jc.syncToEdge(watch.Added, cur)
		},
		DeleteFunc: func(obj interface{}) {
			jc.enqueueController(obj, true)
			jc.syncToEdge(watch.Deleted, obj)
		},
	})
	jc.jobLister = jobInformer.Lister()
	jc.jobStoreSynced = jobInformer.Informer().HasSynced

	podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    jc.addPod,
		UpdateFunc: jc.updatePod,
		DeleteFunc: jc.deletePod,
	})
	jc.podStore = podInformer.Lister()
	jc.podStoreSynced = podInformer.Informer().HasSynced

	return jc, nil
}
