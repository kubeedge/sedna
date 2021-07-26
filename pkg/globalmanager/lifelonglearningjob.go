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

// ljControllerKind contains the schema.GroupVersionKind for this controller type.
var ljControllerKind = sednav1.SchemeGroupVersion.WithKind("LifelongLearningJob")

// LifelongLearningJobController ensures that all LifelongLearningJob objects have corresponding pods to
// run their configured workload.
type LifelongLearningJobController struct {
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

	recorder record.EventRecorder

	cfg *config.ControllerConfig
}

// Run the main goroutine responsible for watching and syncing jobs.
func (jc *LifelongLearningJobController) Start() error {
	workers := 1
	stopCh := messageContext.Done()

	go func() {
		defer utilruntime.HandleCrash()
		defer jc.queue.ShutDown()
		klog.Infof("Starting lifelonglearning job controller")
		defer klog.Infof("Shutting down lifelonglearning job controller")

		if !cache.WaitForNamedCacheSync("lifelonglearningjob", stopCh, jc.podStoreSynced, jc.jobStoreSynced) {
			klog.Errorf("failed to wait for caches to sync")

			return
		}
		klog.Infof("Starting lifelonglearning job workers")
		for i := 0; i < workers; i++ {
			go wait.Until(jc.worker, time.Second, stopCh)
		}

		<-stopCh
	}()
	return nil
}

// enqueueByPod enqueues the lifelonglearningjob object of the specified pod.
func (jc *LifelongLearningJobController) enqueueByPod(pod *v1.Pod, immediate bool) {
	controllerRef := metav1.GetControllerOf(pod)

	if controllerRef == nil {
		return
	}

	if controllerRef.Kind != ljControllerKind.Kind {
		return
	}

	service, err := jc.jobLister.LifelongLearningJobs(pod.Namespace).Get(controllerRef.Name)
	if err != nil {
		return
	}

	if service.UID != controllerRef.UID {
		return
	}

	jc.enqueueController(service, immediate)
}

// When a pod is created, enqueue the controller that manages it and update it's expectations.
func (jc *LifelongLearningJobController) addPod(obj interface{}) {
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

// When a pod is updated, figure out what lifelonglearning job manage it and wake them up.
func (jc *LifelongLearningJobController) updatePod(old, cur interface{}) {
	curPod := cur.(*v1.Pod)
	oldPod := old.(*v1.Pod)

	// no pod update, no queue
	if curPod.ResourceVersion == oldPod.ResourceVersion {
		return
	}

	jc.addPod(curPod)
}

// deletePod enqueues the lifelonglearningjob obj When a pod is deleted
func (jc *LifelongLearningJobController) deletePod(obj interface{}) {
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
	jc.enqueueByPod(pod, true)
}

// obj could be an *sedna.LifelongLearningJob, or a DeletionFinalStateUnknown marker item,
// immediate tells the controller to update the status right away, and should
// happen ONLY when there was a successful pod run.
func (jc *LifelongLearningJobController) enqueueController(obj interface{}, immediate bool) {
	key, err := k8scontroller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for object %+v: %v", obj, err))
		return
	}

	backoff := time.Duration(0)
	if !immediate {
		backoff = getBackoff(jc.queue, key)
	}

	jc.queue.AddAfter(key, backoff)
}

// worker runs a worker thread that just dequeues items, processes them, and marks them done.
// It enforces that the syncHandler is never invoked concurrently with the same key.
func (jc *LifelongLearningJobController) worker() {
	for jc.processNextWorkItem() {
	}
}

func (jc *LifelongLearningJobController) processNextWorkItem() bool {
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

	utilruntime.HandleError(fmt.Errorf("Error syncing lifelonglearning job: %v", err))
	jc.queue.AddRateLimited(key)

	return true
}

// sync will sync the lifelonglearning job with the given key if it has had its expectations fulfilled, meaning
// it did not expect to see any more of its pods created or deleted. This function is not meant to be invoked
// concurrently with the same key.
func (jc *LifelongLearningJobController) sync(key string) (bool, error) {
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
	sharedLifelongLearningJob, err := jc.jobLister.LifelongLearningJobs(ns).Get(name)
	if err != nil {
		if errors.IsNotFound(err) {
			klog.V(4).Infof("lifelonglearning job has been deleted: %v", key)
			return true, nil
		}
		return false, err
	}
	lifelonglearningjob := *sharedLifelongLearningJob
	// set kind for lifelonglearningjob in case that the kind is None
	lifelonglearningjob.SetGroupVersionKind(sednav1.SchemeGroupVersion.WithKind("LifelongLearningJob"))

	// lifelonglearningjob first start
	if lifelonglearningjob.Status.StartTime == nil {
		now := metav1.Now()
		lifelonglearningjob.Status.StartTime = &now
	}

	// if lifelonglearningjob was finished previously, we don't want to redo the termination
	if IsLifelongLearningJobFinished(&lifelonglearningjob) {
		return true, nil
	}

	forget := false
	jobFailed := false
	needUpdated := false

	// update conditions of lifelonglearning job
	needUpdated, err = jc.updateLifelongLearningJobConditions(&lifelonglearningjob)
	if err != nil {
		klog.V(2).Infof("lifelonglearning job %v/%v faied to be updated, err:%s", lifelonglearningjob.Namespace, lifelonglearningjob.Name, err)
	}

	if needUpdated {
		if err := jc.updateLifelongLearningJobStatus(&lifelonglearningjob); err != nil {
			return forget, err
		}

		if jobFailed && !IsLifelongLearningJobFinished(&lifelonglearningjob) {
			// returning an error will re-enqueue LifelongLearningJob after the backoff period
			return forget, fmt.Errorf("failed pod(s) detected for lifelonglearningjob key %q", key)
		}

		forget = true
	}

	return forget, err
}

// updateLifelongLearningJobConditions ensures that conditions of lifelonglearning job can be changed by podstatus
func (jc *LifelongLearningJobController) updateLifelongLearningJobConditions(lifelonglearningjob *sednav1.LifelongLearningJob) (bool, error) {
	var initialType sednav1.LLJobStageConditionType
	var latestCondition sednav1.LLJobCondition = sednav1.LLJobCondition{
		Stage: sednav1.LLJobTrain,
		Type:  initialType,
	}
	var newConditionType sednav1.LLJobStageConditionType
	latestCondition.Stage = sednav1.LLJobTrain
	var needUpdated = false
	jobConditions := lifelonglearningjob.Status.Conditions
	var podStatus v1.PodPhase = v1.PodUnknown
	if len(jobConditions) > 0 {
		// get latest pod and pod status
		latestCondition = (jobConditions)[len(jobConditions)-1]
		klog.V(2).Infof("lifelonglearning job %v/%v latest stage %v:", lifelonglearningjob.Namespace, lifelonglearningjob.Name,
			latestCondition.Stage)
		pod := jc.getSpecifiedPods(lifelonglearningjob, string(latestCondition.Stage))

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
			err = jc.restartInferPod(lifelonglearningjob)
			if err != nil {
				klog.V(2).Infof("lifelonglearning job %v/%v inference pod failed to restart, err:%s", lifelonglearningjob.Namespace, lifelonglearningjob.Name, err)
			} else {
				klog.V(2).Infof("lifelonglearning job %v/%v inference pod restarts successfully", lifelonglearningjob.Namespace, lifelonglearningjob.Name)
			}
		} else if podStatus != v1.PodPending && podStatus != v1.PodRunning {
			err = jc.createPod(lifelonglearningjob, jobStage)
		}
		if err != nil {
			return needUpdated, err
		}
		newConditionType = sednav1.LLJobStageCondStarting

	case sednav1.LLJobStageCondStarting, sednav1.LLJobStageCondRunning:
		if podStatus == v1.PodRunning {
			if jobStage == sednav1.LLJobDeploy {
				newConditionType = sednav1.LLJobStageCondCompleted
			} else {
				// watch pod status, if pod running, set type running
				newConditionType = sednav1.LLJobStageCondRunning
			}
		} else if podStatus == v1.PodSucceeded {
			// watch pod status, if pod completed, set type completed
			newConditionType = sednav1.LLJobStageCondCompleted
			klog.V(2).Infof("lifelonglearning job %v/%v %v stage completed!", lifelonglearningjob.Namespace, lifelonglearningjob.Name, jobStage)
		} else if podStatus == v1.PodFailed {
			newConditionType = sednav1.LLJobStageCondFailed
			klog.V(2).Infof("lifelonglearning job %v/%v %v stage failed!", lifelonglearningjob.Namespace, lifelonglearningjob.Name, jobStage)
		}
	case sednav1.LLJobStageCondCompleted:
		jobStage = jc.getNextStage(jobStage)
		newConditionType = sednav1.LLJobStageCondWaiting

	case sednav1.LLJobStageCondFailed:
		jobStage = sednav1.LLJobTrain
		newConditionType = sednav1.LLJobStageCondWaiting

	default:
		// do nothing when given other type out of cases
	}
	klog.V(2).Infof("lifelonglearning job %v/%v, conditions: %v", lifelonglearningjob.Namespace, lifelonglearningjob.Name, jobConditions)
	if latestCondition.Type != newConditionType {
		lifelonglearningjob.Status.Conditions = append(lifelonglearningjob.Status.Conditions, NewLifelongLearningJobCondition(newConditionType, jobStage))
		needUpdated = true
		return needUpdated, nil
	}
	return needUpdated, nil
}

// updateLifelongLearningJobStatus ensures that jobstatus can be updated rightly
func (jc *LifelongLearningJobController) updateLifelongLearningJobStatus(lifelonglearningjob *sednav1.LifelongLearningJob) error {
	jobClient := jc.client.LifelongLearningJobs(lifelonglearningjob.Namespace)
	var err error
	for i := 0; i <= ResourceUpdateRetries; i = i + 1 {
		var newLifelongLearningJob *sednav1.LifelongLearningJob
		newLifelongLearningJob, err = jobClient.Get(context.TODO(), lifelonglearningjob.Name, metav1.GetOptions{})
		if err != nil {
			break
		}
		newLifelongLearningJob.Status = lifelonglearningjob.Status
		if _, err = jobClient.UpdateStatus(context.TODO(), newLifelongLearningJob, metav1.UpdateOptions{}); err == nil {
			break
		}
	}
	return err
}

func NewLifelongLearningJobCondition(conditionType sednav1.LLJobStageConditionType, jobStage sednav1.LLJobStage) sednav1.LLJobCondition {
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

func (jc *LifelongLearningJobController) generatePodName(jobName string, workerType string) string {
	return jobName + "-" + strings.ToLower(workerType) + "-" + utilrand.String(5)
}

func (jc *LifelongLearningJobController) getSpecifiedPods(job *sednav1.LifelongLearningJob, podType string) *v1.Pod {
	if podType == "Deploy" {
		podType = InferencePodType
	}
	var latestPod *v1.Pod
	selector, _ := GenerateSelector(job)
	pods, err := jc.podStore.Pods(job.Namespace).List(selector)
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

func (jc *LifelongLearningJobController) restartInferPod(job *sednav1.LifelongLearningJob) error {
	inferPod := jc.getSpecifiedPods(job, InferencePodType)
	if inferPod == nil {
		klog.V(2).Infof("No inferpod is running in lifelonglearning job %v/%v", job.Namespace, job.Name)
		err := jc.createInferPod(job)
		return err
	}
	ctx := context.Background()
	err := jc.kubeClient.CoreV1().Pods(job.Namespace).Delete(ctx, inferPod.Name, metav1.DeleteOptions{})
	if err != nil {
		klog.Warningf("failed to delete inference pod %s for lifelonglearning job %v/%v, err:%s", inferPod.Name, job.Namespace, job.Name, err)
		return err
	}
	err = jc.createInferPod(job)
	if err != nil {
		klog.Warningf("failed to create inference pod %s for lifelonglearning job %v/%v, err:%s", inferPod.Name, job.Namespace, job.Name, err)
		return err
	}
	return nil
}

func (jc *LifelongLearningJobController) getNextStage(currentStage sednav1.LLJobStage) sednav1.LLJobStage {
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

func (jc *LifelongLearningJobController) getSecret(namespace, name string, ownerStr string) (secret *v1.Secret, err error) {
	if name != "" {
		secret, err = jc.kubeClient.CoreV1().Secrets(namespace).Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			err = fmt.Errorf("failed to get the secret %s for %s: %w",
				name,
				ownerStr, err)
		}
	}
	return
}

func IsLifelongLearningJobFinished(j *sednav1.LifelongLearningJob) bool {
	// TODO
	return false
}

func (jc *LifelongLearningJobController) createPod(job *sednav1.LifelongLearningJob, podtype sednav1.LLJobStage) (err error) {
	ctx := context.Background()
	var podTemplate *v1.PodTemplateSpec

	LLDatasetName := job.Spec.Dataset.Name

	dataset, err := jc.client.Datasets(job.Namespace).Get(ctx, LLDatasetName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get dataset %s: %w", LLDatasetName, err)
	}

	datasetSecret, err := jc.getSecret(
		job.Namespace,
		dataset.Spec.CredentialName,
		fmt.Sprintf("dataset %s", dataset.Name),
	)
	if err != nil {
		return err
	}

	jobSecret, err := jc.getSecret(
		job.Namespace,
		job.Spec.CredentialName,
		fmt.Sprintf("lifelonglearning job %s", job.Name),
	)
	if err != nil {
		return err
	}

	// get all url for train and eval from data in condition
	condDataStr := job.Status.Conditions[len(job.Status.Conditions)-1].Data
	klog.V(2).Infof("lifelonglearning job %v/%v data condition:%s", job.Namespace, job.Name, condDataStr)
	var cond LifelongLearningCondData
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

	var workerParam *WorkerParam = new(WorkerParam)
	if podtype == sednav1.LLJobTrain {
		workerParam.workerType = "Train"

		podTemplate = &job.Spec.TrainSpec.Template
		// Env parameters for train

		workerParam.env = map[string]string{
			"NAMESPACE":   job.Namespace,
			"JOB_NAME":    job.Name,
			"WORKER_NAME": "train-worker-" + utilrand.String(5),

			"LC_SERVER": jc.cfg.LC.Server,
			"KB_SERVER": jc.cfg.KB.Server,
		}

		workerParam.mounts = append(workerParam.mounts,
			WorkerMount{
				URL: &MountURL{
					URL:                   cond.Input.OutputDir,
					Secret:                jobSecret,
					DownloadByInitializer: false,
				},
				EnvName: "OUTPUT_URL",
			},

			WorkerMount{
				URL: &MountURL{
					URL:                   dataURL,
					Secret:                jobSecret,
					DownloadByInitializer: true,
				},
				EnvName: "TRAIN_DATASET_URL",
			},

			// see https://github.com/kubeedge/sedna/issues/35
			WorkerMount{
				URL: &MountURL{
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
		workerParam.workerType = "Eval"

		// Configure Env information for eval by initial WorkerParam
		workerParam.env = map[string]string{
			"NAMESPACE":   job.Namespace,
			"JOB_NAME":    job.Name,
			"WORKER_NAME": "eval-worker-" + utilrand.String(5),

			"LC_SERVER": jc.cfg.LC.Server,
			"KB_SERVER": jc.cfg.KB.Server,
		}

		var modelMountURLs []MountURL
		for _, url := range inputmodelURLs {
			modelMountURLs = append(modelMountURLs, MountURL{
				URL:                   url,
				Secret:                jobSecret,
				DownloadByInitializer: true,
			})
		}
		workerParam.mounts = append(workerParam.mounts,
			WorkerMount{
				URLs:    modelMountURLs,
				Name:    "models",
				EnvName: "MODEL_URLS",
			},

			WorkerMount{
				URL: &MountURL{
					URL:                   cond.Input.OutputDir,
					Secret:                jobSecret,
					DownloadByInitializer: false,
				},
				EnvName: "OUTPUT_URL",
			},

			WorkerMount{
				URL: &MountURL{
					URL:                   dataURL,
					Secret:                datasetSecret,
					DownloadByInitializer: true,
				},
				Name:    "datasets",
				EnvName: "TEST_DATASET_URL",
			},

			WorkerMount{
				URL: &MountURL{
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
	workerParam.restartPolicy = v1.RestartPolicyOnFailure
	workerParam.hostNetwork = true

	// create pod based on podtype
	_, err = createPodWithTemplate(jc.kubeClient, job, podTemplate, workerParam)
	if err != nil {
		return err
	}
	return
}

func (jc *LifelongLearningJobController) createInferPod(job *sednav1.LifelongLearningJob) error {
	inferModelURL := strings.Join([]string{strings.TrimRight(job.Spec.OutputDir, "/"), "deploy/index.pkl"}, "/")

	jobSecret, err := jc.getSecret(
		job.Namespace,
		job.Spec.CredentialName,
		fmt.Sprintf("lifelonglearning job %s", job.Name),
	)
	if err != nil {
		return err
	}

	var workerParam *WorkerParam = new(WorkerParam)
	workerParam.mounts = append(workerParam.mounts,
		WorkerMount{
			URL: &MountURL{
				URL:                   inferModelURL,
				Secret:                jobSecret,
				DownloadByInitializer: false,
			},
			Name:    "models",
			EnvName: "MODEL_URLS",
		},
	)

	workerParam.env = map[string]string{
		"NAMESPACE":   job.Namespace,
		"JOB_NAME":    job.Name,
		"WORKER_NAME": "inferworker-" + utilrand.String(5),

		"LC_SERVER": jc.cfg.LC.Server,
	}

	workerParam.workerType = InferencePodType
	workerParam.hostNetwork = true

	// create edge pod
	_, err = createPodWithTemplate(jc.kubeClient, job, &job.Spec.DeploySpec.Template, workerParam)
	return err
}

// GetName returns the name of the lifelonglearning job controller
func (jc *LifelongLearningJobController) GetName() string {
	return "LifelongLearningJobController"
}

// NewLifelongLearningJobController creates a new LifelongLearningJob controller that keeps the relevant pods
// in sync with their corresponding LifelongLearningJob objects.
func NewLifelongLearningJobController(cfg *config.ControllerConfig) (FeatureControllerI, error) {
	namespace := cfg.Namespace
	if namespace == "" {
		namespace = metav1.NamespaceAll
	}
	kubeClient, err := utils.KubeClient()
	if err != nil {
		return nil, err
	}

	kubecfg, err := utils.KubeConfig()
	if err != nil {
		return nil, err
	}
	crdclient, err := clientset.NewForConfig(kubecfg)
	if err != nil {
		return nil, err
	}

	kubeInformerFactory := kubeinformers.NewSharedInformerFactoryWithOptions(kubeClient, time.Second*30, kubeinformers.WithNamespace(namespace))

	podInformer := kubeInformerFactory.Core().V1().Pods()

	jobInformerFactory := informers.NewSharedInformerFactoryWithOptions(crdclient, time.Second*30, informers.WithNamespace(namespace))
	jobInformer := jobInformerFactory.Sedna().V1alpha1().LifelongLearningJobs()

	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: kubeClient.CoreV1().Events("")})

	jc := &LifelongLearningJobController{
		kubeClient: kubeClient,
		client:     crdclient.SednaV1alpha1(),
		queue:      workqueue.NewNamedRateLimitingQueue(workqueue.NewItemExponentialFailureRateLimiter(DefaultBackOff, MaxBackOff), "lifelonglearningjob"),
		recorder:   eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "lifelonglearningjob-controller"}),
		cfg:        cfg,
	}

	jobInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
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
	jc.jobLister = jobInformer.Lister()
	jc.jobStoreSynced = jobInformer.Informer().HasSynced

	podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    jc.addPod,
		UpdateFunc: jc.updatePod,
		DeleteFunc: jc.deletePod,
	})
	jc.podStore = podInformer.Lister()
	jc.podStoreSynced = podInformer.Informer().HasSynced

	stopCh := make(chan struct{})
	kubeInformerFactory.Start(stopCh)
	jobInformerFactory.Start(stopCh)
	return jc, err
}
