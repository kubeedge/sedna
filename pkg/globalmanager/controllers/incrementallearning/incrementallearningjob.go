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

package incrementallearning

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilrand "k8s.io/apimachinery/pkg/util/rand"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
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
	// Name is this controller name
	Name = "IncrementalLearning"

	// KindName is the kind name of CR this controller controls
	KindName = "IncrementalLearningJob"
)

// Kind contains the schema.GroupVersionKind for this controller type.
var Kind = sednav1.SchemeGroupVersion.WithKind(KindName)

// Controller ensures that all IncrementalLearningJob objects have corresponding pods to
// run their configured workload.
type Controller struct {
	kubeClient kubernetes.Interface
	client     sednaclientset.SednaV1alpha1Interface

	// podStoreSynced returns true if the pod store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	podStoreSynced cache.InformerSynced
	// jobStoreSynced returns true if the incrementaljob store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	jobStoreSynced cache.InformerSynced

	// A store of jobs
	jobLister sednav1listers.IncrementalLearningJobLister

	// A store of pods, populated by the podController
	podStore corelisters.PodLister

	// IncrementalLearningJobs that need to be updated
	queue workqueue.RateLimitingInterface

	recorder record.EventRecorder

	cfg *config.ControllerConfig
}

// Run starts the main goroutine responsible for watching and syncing jobs.
func (c *Controller) Run(stopCh <-chan struct{}) {
	// TODO: make workers parameter
	workers := 1

	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()

	klog.Infof("Starting %s controller", Name)
	defer klog.Infof("Shutting down %s controller", Name)

	if !cache.WaitForNamedCacheSync(Name, stopCh, c.podStoreSynced, c.jobStoreSynced) {
		klog.Errorf("failed to wait for %s caches to sync", Name)

		return
	}
	klog.Infof("Starting %s job workers", Name)
	for i := 0; i < workers; i++ {
		go wait.Until(c.worker, time.Second, stopCh)
	}

	<-stopCh
}

// enqueueByPod enqueues the jointInferenceService object of the specified pod.
func (c *Controller) enqueueByPod(pod *v1.Pod, immediate bool) {
	controllerRef := metav1.GetControllerOf(pod)

	if controllerRef == nil {
		return
	}

	if controllerRef.Kind != Kind.Kind {
		return
	}

	service, err := c.jobLister.IncrementalLearningJobs(pod.Namespace).Get(controllerRef.Name)
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

// deletePod enqueues the jointinferenceservice obj When a pod is deleted
func (c *Controller) deletePod(obj interface{}) {
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
	c.enqueueByPod(pod, true)
}

// obj could be an *sedna.IncrementalLearningJob, or a DeletionFinalStateUnknown marker item,
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

	utilruntime.HandleError(fmt.Errorf("Error syncing incrementallearning job: %v", err))
	c.queue.AddRateLimited(key)

	return true
}

// sync will sync the incrementallearning job with the given key if it has had its expectations fulfilled, meaning
// it did not expect to see any more of its pods created or deleted. This function is not meant to be invoked
// concurrently with the same key.
func (c *Controller) sync(key string) (bool, error) {
	startTime := time.Now()
	defer func() {
		klog.V(4).Infof("Finished syncing incrementallearning job %q (%v)", key, time.Since(startTime))
	}()

	ns, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return false, err
	}
	if len(ns) == 0 || len(name) == 0 {
		return false, fmt.Errorf("invalid incrementallearning job key %q: either namespace or name is missing", key)
	}
	sharedIncrementalJob, err := c.jobLister.IncrementalLearningJobs(ns).Get(name)
	if err != nil {
		if errors.IsNotFound(err) {
			klog.V(4).Infof("incrementallearning job has been deleted: %v", key)
			return true, nil
		}
		return false, err
	}
	incrementaljob := *sharedIncrementalJob
	// set kind for incrementaljob in case that the kind is None
	incrementaljob.SetGroupVersionKind(sednav1.SchemeGroupVersion.WithKind("IncrementalLearningJob"))
	// incrementaljob first start, create pod for inference
	if incrementaljob.Status.StartTime == nil {
		now := metav1.Now()
		incrementaljob.Status.StartTime = &now
		pod := c.getSpecifiedPods(&incrementaljob, runtime.InferencePodType)
		if pod == nil {
			err = c.createInferPod(&incrementaljob)
		} else {
			if pod.Status.Phase != v1.PodRunning && pod.Status.Phase != v1.PodPending {
				err = c.createInferPod(&incrementaljob)
			}
		}
		if err != nil {
			return false, nil
		}
	}

	// if incrementaljob was finished previously, we don't want to redo the termination
	if IsIncrementalJobFinished(&incrementaljob) {
		return true, nil
	}

	forget := false
	jobFailed := false
	needUpdated := false

	// update conditions of incremental job
	needUpdated, err = c.updateIncrementalJobConditions(&incrementaljob)
	if err != nil {
		klog.V(2).Infof("incrementallearning job %v/%v faied to be updated, err:%s", incrementaljob.Namespace, incrementaljob.Name, err)
	}

	if needUpdated {
		if err := c.updateIncrementalJobStatus(&incrementaljob); err != nil {
			return forget, err
		}

		if jobFailed && !IsIncrementalJobFinished(&incrementaljob) {
			// returning an error will re-enqueue IncrementalJob after the backoff period
			return forget, fmt.Errorf("failed pod(s) detected for incrementaljob key %q", key)
		}

		forget = true
	}

	return forget, err
}

// setWorkerNodeNameOfJob sets the worker nodeName of the specified job
// which is used for downstream to sync job info to the specified LC located in nodeName.
func (c *Controller) setWorkerNodeNameOfJob(job *sednav1.IncrementalLearningJob, jobStage string, nodeName string) error {
	key := runtime.AnnotationsKeyPrefix + jobStage

	ann := job.GetAnnotations()
	if ann != nil {
		if ann[key] == nodeName {
			// already set
			return nil
		}
	}

	jobClient := c.client.IncrementalLearningJobs(job.Namespace)
	var err error
	for i := 0; i <= runtime.ResourceUpdateRetries; i++ {
		var newJob *sednav1.IncrementalLearningJob
		newJob, err = jobClient.Get(context.TODO(), job.Name, metav1.GetOptions{})
		if err != nil {
			break
		}

		annotations := newJob.GetAnnotations()
		if annotations != nil {
			if annotations[key] == nodeName {
				return nil
			}
		}

		dataStr := fmt.Sprintf(`{"metadata":{"annotations":{"%s":"%s"}}}`, key, nodeName)
		if _, err = jobClient.Patch(context.TODO(), job.Name, types.MergePatchType, []byte(dataStr), metav1.PatchOptions{}); err == nil {
			break
		}
	}

	return err
}

// updateIncrementalJobConditions ensures that conditions of incrementallearning job can be changed by podstatus
func (c *Controller) updateIncrementalJobConditions(incrementaljob *sednav1.IncrementalLearningJob) (bool, error) {
	var initialType sednav1.ILJobStageConditionType
	var latestCondition sednav1.ILJobCondition = sednav1.ILJobCondition{
		Stage: sednav1.ILJobTrain,
		Type:  initialType,
	}
	var newConditionType sednav1.ILJobStageConditionType
	var needUpdated = false
	jobConditions := incrementaljob.Status.Conditions
	var podStatus v1.PodPhase = v1.PodUnknown
	var pod *v1.Pod
	if len(jobConditions) > 0 {
		// get latest pod and pod status
		latestCondition = (jobConditions)[len(jobConditions)-1]
		klog.V(2).Infof("incrementallearning job %v/%v latest stage %v:", incrementaljob.Namespace, incrementaljob.Name,
			latestCondition.Stage)
		pod = c.getSpecifiedPods(incrementaljob, string(latestCondition.Stage))

		if pod != nil {
			podStatus = pod.Status.Phase
		}
	}
	jobStage := latestCondition.Stage
	currentType := latestCondition.Type
	newConditionType = currentType

	switch currentType {
	case initialType:
		newConditionType = sednav1.ILJobStageCondWaiting

	case sednav1.ILJobStageCondWaiting:
		// do nothing, waiting for LC to set type from waiting to ready

	case sednav1.ILJobStageCondReady:
		// create a pod, and set type from ready to starting
		// include train, eval, deploy pod
		var err error
		if jobStage == sednav1.ILJobDeploy {
			err = c.restartInferPod(incrementaljob)
			if err != nil {
				klog.V(2).Infof("incrementallearning job %v/%v inference pod failed to restart, err:%s", incrementaljob.Namespace, incrementaljob.Name, err)
			} else {
				klog.V(2).Infof("incrementallearning job %v/%v inference pod restarts successfully", incrementaljob.Namespace, incrementaljob.Name)
			}
		} else if podStatus != v1.PodPending && podStatus != v1.PodRunning {
			err = c.createPod(incrementaljob, jobStage)
		}
		if err != nil {
			return needUpdated, err
		}
		newConditionType = sednav1.ILJobStageCondStarting

	case sednav1.ILJobStageCondStarting, sednav1.ILJobStageCondRunning:
		if podStatus == v1.PodRunning {
			if jobStage == sednav1.ILJobDeploy {
				newConditionType = sednav1.ILJobStageCondCompleted
			} else {
				// watch pod status, if pod running, set type running
				newConditionType = sednav1.ILJobStageCondRunning

				// add nodeName to job
				if err := c.setWorkerNodeNameOfJob(incrementaljob, string(jobStage), pod.Spec.NodeName); err != nil {
					return needUpdated, err
				}
			}
		} else if podStatus == v1.PodSucceeded {
			// watch pod status, if pod completed, set type completed
			newConditionType = sednav1.ILJobStageCondCompleted
			klog.V(2).Infof("incrementallearning job %v/%v %v stage completed!", incrementaljob.Namespace, incrementaljob.Name, jobStage)
		} else if podStatus == v1.PodFailed {
			newConditionType = sednav1.ILJobStageCondFailed
			klog.V(2).Infof("incrementallearning job %v/%v %v stage failed!", incrementaljob.Namespace, incrementaljob.Name, jobStage)
		}
	case sednav1.ILJobStageCondCompleted:
		jobStage = getNextStage(jobStage)
		newConditionType = sednav1.ILJobStageCondWaiting

	case sednav1.ILJobStageCondFailed:
		jobStage = sednav1.ILJobTrain
		newConditionType = sednav1.ILJobStageCondWaiting

	default:
		// do nothing when given other type out of cases
	}
	klog.V(2).Infof("incrementallearning job %v/%v, conditions: %v", incrementaljob.Namespace, incrementaljob.Name, jobConditions)
	if latestCondition.Type != newConditionType {
		incrementaljob.Status.Conditions = append(incrementaljob.Status.Conditions, NewIncrementalJobCondition(newConditionType, jobStage))
		needUpdated = true
		return needUpdated, nil
	}
	return needUpdated, nil
}

// updateIncrementalJobStatus ensures that jobstatus can be updated rightly
func (c *Controller) updateIncrementalJobStatus(incrementaljob *sednav1.IncrementalLearningJob) error {
	jobClient := c.client.IncrementalLearningJobs(incrementaljob.Namespace)
	var err error
	for i := 0; i <= runtime.ResourceUpdateRetries; i++ {
		var newIncrementalJob *sednav1.IncrementalLearningJob
		newIncrementalJob, err = jobClient.Get(context.TODO(), incrementaljob.Name, metav1.GetOptions{})
		if err != nil {
			break
		}
		newIncrementalJob.Status = incrementaljob.Status
		if _, err = jobClient.UpdateStatus(context.TODO(), newIncrementalJob, metav1.UpdateOptions{}); err == nil {
			break
		}
	}
	return err
}

func NewIncrementalJobCondition(conditionType sednav1.ILJobStageConditionType, jobStage sednav1.ILJobStage) sednav1.ILJobCondition {
	return sednav1.ILJobCondition{
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

func (c *Controller) getSpecifiedPods(job *sednav1.IncrementalLearningJob, podType string) *v1.Pod {
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

func (c *Controller) restartInferPod(job *sednav1.IncrementalLearningJob) error {
	inferPod := c.getSpecifiedPods(job, runtime.InferencePodType)
	if inferPod == nil {
		klog.V(2).Infof("No inferpod is running in incrementallearning job %v/%v", job.Namespace, job.Name)
		err := c.createInferPod(job)
		return err
	}
	ctx := context.Background()
	err := c.kubeClient.CoreV1().Pods(job.Namespace).Delete(ctx, inferPod.Name, metav1.DeleteOptions{})
	if err != nil {
		klog.Warningf("failed to delete inference pod %s for incrementallearning job %v/%v, err:%s", inferPod.Name, job.Namespace, job.Name, err)
		return err
	}
	err = c.createInferPod(job)
	if err != nil {
		klog.Warningf("failed to create inference pod %s for incrementallearning job %v/%v, err:%s", inferPod.Name, job.Namespace, job.Name, err)
		return err
	}
	return nil
}

func getNextStage(currentStage sednav1.ILJobStage) sednav1.ILJobStage {
	switch currentStage {
	case sednav1.ILJobTrain:
		return sednav1.ILJobEval
	case sednav1.ILJobEval:
		return sednav1.ILJobDeploy
	case sednav1.ILJobDeploy:
		return sednav1.ILJobTrain
	default:
		return sednav1.ILJobTrain
	}
}

func IsIncrementalJobFinished(j *sednav1.IncrementalLearningJob) bool {
	// TODO
	return false
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

func (c *Controller) createPod(job *sednav1.IncrementalLearningJob, podtype sednav1.ILJobStage) (err error) {
	ctx := context.Background()
	var podTemplate *v1.PodTemplateSpec

	incrementalDatasetName := job.Spec.Dataset.Name
	initialModelName := job.Spec.InitialModel.Name
	deployModelName := job.Spec.DeploySpec.Model.Name

	// check initial model name
	initialModel, err := c.client.Models(job.Namespace).Get(ctx, initialModelName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get initial model %s: %w",
			initialModelName, err)
	}

	_, err = c.client.Models(job.Namespace).Get(ctx, deployModelName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get deploy model %s: %w",
			deployModelName, err)
	}

	dataset, err := c.client.Datasets(job.Namespace).Get(ctx, incrementalDatasetName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get dataset %s: %w",
			incrementalDatasetName, err)
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
		fmt.Sprintf("incremental job %s", job.Name),
	)
	if err != nil {
		return err
	}

	// get all url for train and eval from data in condition
	condDataStr := job.Status.Conditions[len(job.Status.Conditions)-1].Data
	klog.V(2).Infof("incrementallearning job %v/%v data condition:%s", job.Namespace, job.Name, condDataStr)
	var cond IncrementalCondData
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
	if podtype == sednav1.ILJobTrain {
		workerParam.WorkerType = runtime.TrainPodType

		podTemplate = &job.Spec.TrainSpec.Template
		// Env parameters for train

		workerParam.Env = map[string]string{
			"NAMESPACE":   job.Namespace,
			"JOB_NAME":    job.Name,
			"WORKER_NAME": "train-worker-" + utilrand.String(5),

			"LC_SERVER": c.cfg.LC.Server,
		}

		baseModelURL := inputmodelURLs[0]
		var baseModelSecret *v1.Secret
		if baseModelURL == initialModel.Spec.URL {
			baseModelSecret, err = c.getSecret(
				job.Namespace,
				initialModel.Spec.CredentialName,
				fmt.Sprintf("initial model %s", initialModelName),
			)
			if err != nil {
				return err
			}
		} else {
			baseModelSecret = jobSecret
		}

		workerParam.Mounts = append(workerParam.Mounts,
			runtime.WorkerMount{
				URL: &runtime.MountURL{
					URL:                   baseModelURL,
					Secret:                baseModelSecret,
					DownloadByInitializer: true,
				},
				EnvName: "BASE_MODEL_URL",
			},
			runtime.WorkerMount{
				URL: &runtime.MountURL{
					URL:                   cond.Input.OutputDir,
					Secret:                jobSecret,
					DownloadByInitializer: false,
				},
				EnvName: "MODEL_URL",
			},

			runtime.WorkerMount{
				URL: &runtime.MountURL{
					URL:                   dataURL,
					DownloadByInitializer: true,
					Secret:                jobSecret,
				},
				EnvName: "TRAIN_DATASET_URL",
			},

			// see https://github.com/kubeedge/sedna/issues/35
			runtime.WorkerMount{
				URL: &runtime.MountURL{
					Secret:                datasetSecret,
					URL:                   originalDataURLOrIndex,
					DownloadByInitializer: true,
					Indirect:              dataset.Spec.URL != originalDataURLOrIndex,
				},
				EnvName: "ORIGINAL_DATASET_URL",
			},
		)
	} else {
		podTemplate = &job.Spec.EvalSpec.Template
		workerParam.WorkerType = "Eval"

		// Configure Env information for eval by initial runtime.WorkerParam
		workerParam.Env = map[string]string{
			"NAMESPACE":   job.Namespace,
			"JOB_NAME":    job.Name,
			"WORKER_NAME": "eval-worker-" + utilrand.String(5),

			"LC_SERVER": c.cfg.LC.Server,
		}

		var modelMountURLs []runtime.MountURL
		for _, url := range inputmodelURLs {
			var modelSecret *v1.Secret
			if url == initialModel.Spec.URL {
				modelSecret, err = c.getSecret(
					job.Namespace,
					initialModel.Spec.CredentialName,
					fmt.Sprintf("initial model %s", initialModelName),
				)
				if err != nil {
					return err
				}
			} else {
				modelSecret = jobSecret
			}

			modelMountURLs = append(modelMountURLs, runtime.MountURL{
				URL:                   url,
				Secret:                modelSecret,
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

	// create pod based on podtype
	_, err = runtime.CreatePodWithTemplate(c.kubeClient, job, podTemplate, workerParam)
	if err != nil {
		return err
	}
	return
}

func (c *Controller) createInferPod(job *sednav1.IncrementalLearningJob) error {
	infermodelName := job.Spec.DeploySpec.Model.Name
	inferModel, err := c.client.Models(job.Namespace).Get(context.TODO(), infermodelName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get infer model %s: %w",
			infermodelName, err)
	}
	inferModelURL := inferModel.Spec.URL

	// Env parameters for edge
	HEMParameterJSON, _ := json.Marshal(job.Spec.DeploySpec.HardExampleMining.Parameters)
	HEMParameterString := string(HEMParameterJSON)

	// Configure container mounting and Env information by initial runtime.WorkerParam
	modelSecret, err := c.getSecret(
		job.Namespace,
		inferModel.Spec.CredentialName,
		fmt.Sprintf("model %s", inferModel.Name),
	)
	var workerParam *runtime.WorkerParam = new(runtime.WorkerParam)
	workerParam.Mounts = append(workerParam.Mounts,
		runtime.WorkerMount{
			URL: &runtime.MountURL{
				URL:                   inferModelURL,
				Secret:                modelSecret,
				DownloadByInitializer: true,
			},
			Name:    "model",
			EnvName: "MODEL_URL",
		},
	)

	workerParam.Env = map[string]string{
		"NAMESPACE":   job.Namespace,
		"JOB_NAME":    job.Name,
		"WORKER_NAME": "inferworker-" + utilrand.String(5),

		"HEM_NAME":       job.Spec.DeploySpec.HardExampleMining.Name,
		"HEM_PARAMETERS": HEMParameterString,

		"LC_SERVER": c.cfg.LC.Server,
	}

	workerParam.WorkerType = runtime.InferencePodType
	workerParam.HostNetwork = true

	// create edge pod
	_, err = runtime.CreatePodWithTemplate(c.kubeClient, job, &job.Spec.DeploySpec.Template, workerParam)
	return err
}

// New creates a new IncrementalJob controller that keeps the relevant pods
// in sync with their corresponding IncrementalJob objects.
func New(cc *runtime.ControllerContext) (runtime.FeatureControllerI, error) {
	podInformer := cc.KubeInformerFactory.Core().V1().Pods()

	jobInformer := cc.SednaInformerFactory.Sedna().V1alpha1().IncrementalLearningJobs()

	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: cc.KubeClient.CoreV1().Events("")})

	jc := &Controller{
		kubeClient: cc.KubeClient,
		client:     cc.SednaClient.SednaV1alpha1(),

		queue:    workqueue.NewNamedRateLimitingQueue(workqueue.NewItemExponentialFailureRateLimiter(runtime.DefaultBackOff, runtime.MaxBackOff), "incrementallearningjob"),
		recorder: eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "incrementallearningjob-controller"}),
		cfg:      cc.Config,
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

	jc.addUpstreamHandler(cc)

	return jc, nil
}
