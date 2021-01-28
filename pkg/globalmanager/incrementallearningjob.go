package globalmanager

import (
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"
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

// ijControllerKind contains the schema.GroupVersionKind for this controller type.
var ijControllerKind = sednav1.SchemeGroupVersion.WithKind("IncrementalLearningJob")

// IncrementalJobController ensures that all IncrementalLearningJob objects have corresponding pods to
// run their configured workload.
type IncrementalJobController struct {
	kubeClient kubernetes.Interface
	client     sednaclientset.SednaV1alpha1Interface
	podControl k8scontroller.PodControlInterface

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

// Run the main goroutine responsible for watching and syncing jobs.
func (jc *IncrementalJobController) Start() error {
	workers := 1
	stopCh := messageContext.Done()

	go func() {
		defer utilruntime.HandleCrash()
		defer jc.queue.ShutDown()
		klog.Infof("Starting incrementallearning job controller")
		defer klog.Infof("Shutting down incrementallearning job controller")

		if !cache.WaitForNamedCacheSync("incrementallearningjob", stopCh, jc.podStoreSynced, jc.jobStoreSynced) {
			klog.Errorf("failed to wait for caches to sync")

			return
		}
		klog.Infof("Starting incrementallearning job workers")
		for i := 0; i < workers; i++ {
			go wait.Until(jc.worker, time.Second, stopCh)
		}

		<-stopCh
	}()
	return nil
}

// enqueueByPod enqueues the jointInferenceService object of the specified pod.
func (jc *IncrementalJobController) enqueueByPod(pod *v1.Pod, immediate bool) {
	controllerRef := metav1.GetControllerOf(pod)

	if controllerRef == nil {
		return
	}

	if controllerRef.Kind != jointServiceControllerKind.Kind {
		return
	}

	service, err := jc.jobLister.IncrementalLearningJobs(pod.Namespace).Get(controllerRef.Name)
	if err != nil {
		return
	}

	if service.UID != controllerRef.UID {
		return
	}

	jc.enqueueController(service, immediate)
}

// When a pod is created, enqueue the controller that manages it and update it's expectations.
func (jc *IncrementalJobController) addPod(obj interface{}) {
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
func (jc *IncrementalJobController) updatePod(old, cur interface{}) {
	curPod := cur.(*v1.Pod)
	oldPod := old.(*v1.Pod)

	// no pod update, no queue
	if curPod.ResourceVersion == oldPod.ResourceVersion {
		return
	}

	jc.addPod(curPod)
}

// deletePod enqueues the jointinferenceservice obj When a pod is deleted
func (jc *IncrementalJobController) deletePod(obj interface{}) {
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

// obj could be an *sedna.IncrementalLearningJob, or a DeletionFinalStateUnknown marker item,
// immediate tells the controller to update the status right away, and should
// happen ONLY when there was a successful pod run.
func (jc *IncrementalJobController) enqueueController(obj interface{}, immediate bool) {
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
func (jc *IncrementalJobController) worker() {
	for jc.processNextWorkItem() {
	}
}

func (jc *IncrementalJobController) processNextWorkItem() bool {
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

	utilruntime.HandleError(fmt.Errorf("Error syncing incrementallearning job: %v", err))
	jc.queue.AddRateLimited(key)

	return true
}

// sync will sync the incrementallearning job with the given key if it has had its expectations fulfilled, meaning
// it did not expect to see any more of its pods created or deleted. This function is not meant to be invoked
// concurrently with the same key.
func (jc *IncrementalJobController) sync(key string) (bool, error) {
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
	sharedIncrementalJob, err := jc.jobLister.IncrementalLearningJobs(ns).Get(name)
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
		pod := jc.getSpecifiedPods(&incrementaljob, "inference")
		if pod == nil {
			err = jc.createInferPod(&incrementaljob)
		} else {
			if pod.Status.Phase != v1.PodRunning && pod.Status.Phase != v1.PodPending {
				err = jc.createInferPod(&incrementaljob)
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
	needUpdated, err = jc.updateIncrementalJobConditions(&incrementaljob)
	if err != nil {
		klog.V(2).Infof("incrementallearning job %v/%v faied to be updated, err:%s", incrementaljob.Namespace, incrementaljob.Name, err)
	}

	if needUpdated {
		if err := jc.updateIncrementalJobStatus(&incrementaljob); err != nil {
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

// updateIncrementalJobConditions ensures that conditions of incrementallearning job can be changed by podstatus
func (jc *IncrementalJobController) updateIncrementalJobConditions(incrementaljob *sednav1.IncrementalLearningJob) (bool, error) {
	var initialType sednav1.ILJobStageConditionType
	var latestCondition sednav1.ILJobCondition = sednav1.ILJobCondition{
		Stage: sednav1.ILJobTrain,
		Type:  initialType,
	}
	var newConditionType sednav1.ILJobStageConditionType
	latestCondition.Stage = sednav1.ILJobTrain
	var needUpdated = false
	jobConditions := incrementaljob.Status.Conditions
	var podStatus v1.PodPhase = v1.PodUnknown
	if len(jobConditions) > 0 {
		// get latest pod and pod status
		latestCondition = (jobConditions)[len(jobConditions)-1]
		klog.V(2).Infof("incrementallearning job %v/%v latest stage %v:", incrementaljob.Namespace, incrementaljob.Name,
			latestCondition.Stage)
		pod := jc.getSpecifiedPods(incrementaljob, string(latestCondition.Stage))

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
			err = jc.restartInferPod(incrementaljob)
			if err != nil {
				klog.V(2).Infof("incrementallearning job %v/%v inference pod failed to restart, err:%s", incrementaljob.Namespace, incrementaljob.Name, err)
			} else {
				klog.V(2).Infof("incrementallearning job %v/%v inference pod restarts successfully", incrementaljob.Namespace, incrementaljob.Name)
			}
		} else if podStatus != v1.PodPending && podStatus != v1.PodRunning {
			err = jc.createPod(incrementaljob, jobStage)
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
func (jc *IncrementalJobController) updateIncrementalJobStatus(incrementaljob *sednav1.IncrementalLearningJob) error {
	jobClient := jc.client.IncrementalLearningJobs(incrementaljob.Namespace)
	var err error
	for i := 0; i <= statusUpdateRetries; i = i + 1 {
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

func (jc *IncrementalJobController) generatePodName(jobName string, workerType string) string {
	return jobName + "-" + strings.ToLower(workerType) + "-" + utilrand.String(5)
}

func (jc *IncrementalJobController) getSpecifiedPods(job *sednav1.IncrementalLearningJob, podType string) *v1.Pod {
	if podType == "Deploy" {
		podType = "inference"
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

func (jc *IncrementalJobController) restartInferPod(job *sednav1.IncrementalLearningJob) error {
	inferPod := jc.getSpecifiedPods(job, "inference")
	if inferPod == nil {
		klog.V(2).Infof("No inferpod is running in incrementallearning job %v/%v", job.Namespace, job.Name)
		err := jc.createInferPod(job)
		return err
	}
	ctx := context.Background()
	err := jc.kubeClient.CoreV1().Pods(job.Namespace).Delete(ctx, inferPod.Name, metav1.DeleteOptions{})
	if err != nil {
		klog.Warningf("failed to delete inference pod %s for incrementallearning job %v/%v, err:%s", inferPod.Name, job.Namespace, job.Name, err)
		return err
	}
	err = jc.createInferPod(job)
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

func (jc *IncrementalJobController) createPod(job *sednav1.IncrementalLearningJob, podtype sednav1.ILJobStage) (err error) {
	ctx := context.Background()
	var workerName string
	var workerSpec sednav1.CommonWorkerSpec
	trainworkerspec := job.Spec.TrainSpec
	evalworkerspec := job.Spec.EvalSpec

	incrementalDatasetName := job.Spec.Dataset.Name
	initialModelName := job.Spec.InitialModel.Name
	deployModelName := job.Spec.DeploySpec.Model.Name

	// get basemodel URL, deploymodel, dataset URL
	var basemodelPath string
	var deploymodelPath string
	var datasetPath string

	basemodel, err := jc.client.Models(job.Namespace).Get(ctx, initialModelName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get initial model %s: %w",
			initialModelName, err)
	}
	basemodelPath = filepath.Dir(basemodel.Spec.URL)

	deploymodel, err := jc.client.Models(job.Namespace).Get(ctx, deployModelName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get deploy model %s: %w",
			deployModelName, err)
	}
	deploymodelPath = filepath.Dir(deploymodel.Spec.URL)

	dataset, err := jc.client.Datasets(job.Namespace).Get(ctx, incrementalDatasetName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get dataset %s: %w",
			incrementalDatasetName, err)
	}
	datasetPath = dataset.Spec.URL

	outputDir := job.Spec.OutputDir
	datasetParent := filepath.Dir(datasetPath)
	var trainCodePath string
	var evalCodePath string
	trainCodePath = trainworkerspec.WorkerSpec.ScriptDir
	evalCodePath = evalworkerspec.WorkerSpec.ScriptDir

	trainParameterJSON, _ := json.Marshal(trainworkerspec.WorkerSpec.Parameters)
	evalParameterJSON, _ := json.Marshal(evalworkerspec.WorkerSpec.Parameters)
	trainParameterString := string(trainParameterJSON)
	evalParameterString := string(evalParameterJSON)

	klog.V(2).Infof("incrementallearning job %v/%v train parameters:%s", job.Namespace, job.Name, trainParameterString)
	klog.V(2).Infof("incrementallearning job %v/%v eval parameters:%s", job.Namespace, job.Name, evalParameterString)

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
	outputmodelURL := cond.Input.OutputDir

	// convert user inputs into its form in the container
	var inputmodelURLList []string
	var inputmodelURLContain string
	var outputmodelURLContain string
	var dataURLContain string

	// process inputmodelURLs, add dataPrefix to ench inputmodelURL, return inputmodelURLList
	for _, URL := range inputmodelURLs {
		inputmodelURLList = append(inputmodelURLList, dataPrefix+URL)
	}

	// three container Url for data, inputmodel, outputmodel
	inputmodelURLContain = strings.Join(inputmodelURLList, ";")
	outputmodelURLContain = dataPrefix + outputmodelURL
	dataURLContain = dataPrefix + dataURL

	// Container VolumeMounts parameters
	codeConPath := codePrefix
	dataConPath := dataPrefix + datasetParent
	basemodelConPath := dataPrefix + basemodelPath
	deploymodelConPath := dataPrefix + deploymodelPath
	outputConPath := dataPrefix + outputDir
	var containerPara *ContainerPara = new(ContainerPara)
	if podtype == sednav1.ILJobTrain {
		workerName = "Train"
		workerSpec = trainworkerspec.WorkerSpec
		// Env parameters for train
		preModelURL := inputmodelURLContain     // premodel savepath before increase
		outputModelURL := outputmodelURLContain // outputmodel savepath after increase, should be under outputdir
		trainDataURL := dataURLContain

		// Configure container mounting and Env information for train by initial ContainerPara
		containerPara.volumeMountList = []string{codeConPath, dataConPath, basemodelConPath, deploymodelConPath, outputConPath}
		containerPara.volumeList = []string{trainCodePath, datasetParent, basemodelPath, deploymodelPath, outputDir}
		containerPara.volumeMapName = []string{"code", "data", "base-model", "deploy-model", "output-dir"}
		containerPara.env = map[string]string{
			"TRAIN_DATASET_URL": trainDataURL,
			"MODEL_URL":         outputModelURL,
			"BASE_MODEL_URL":    preModelURL,
			"NAMESPACE":         job.Namespace,
			"JOB_NAME":          job.Name,
			"WORKER_NAME":       "train-worker-" + utilrand.String(5),
			"PARAMETERS":        trainParameterString,
			"LC_SERVER":         jc.cfg.LC.Server,
		}
	} else {
		workerName = "Eval"
		workerSpec = evalworkerspec.WorkerSpec
		// Env parameters for eval
		evalDataURL := dataURLContain
		modelForEval := inputmodelURLContain // can be single or multi models

		// Configure container mounting and Env information for eval by initial ContainerPara
		containerPara.volumeMountList = []string{codeConPath, dataConPath, basemodelConPath, deploymodelConPath, outputConPath}
		containerPara.volumeList = []string{evalCodePath, datasetParent, basemodelPath, deploymodelPath, outputDir}
		containerPara.volumeMapName = []string{"code", "data", "base-model", "deploy-model", "output-dir"}
		containerPara.env = map[string]string{
			"TEST_DATASET_URL": evalDataURL,
			"MODEL_URLS":       modelForEval,
			"NAMESPACE":        job.Namespace,
			"JOB_NAME":         job.Name,
			"WORKER_NAME":      "eval-worker-" + utilrand.String(5),
			"PARAMETERS":       evalParameterString,
			"LC_SERVER":        jc.cfg.LC.Server,
		}
	}
	// create pod based on podtype
	err = jc.generatePod(job, workerSpec, workerName, containerPara)
	if err != nil {
		return err
	}
	return
}

func (jc *IncrementalJobController) createInferPod(job *sednav1.IncrementalLearningJob) error {
	ctx := context.Background()
	infermodelName := job.Spec.DeploySpec.Model.Name
	inferModel, err := jc.client.Models(job.Namespace).Get(ctx, infermodelName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get infer model %s: %w",
			infermodelName, err)
	}
	inferModelPath := inferModel.Spec.URL

	// convert crd to JSON, and put them into env of container
	inferModelParent := filepath.Dir(inferModelPath)

	inferWorkerSpec := job.Spec.DeploySpec.WorkerSpec
	inferCodePath := inferWorkerSpec.ScriptDir
	inferParameterJSON, _ := json.Marshal(inferWorkerSpec.Parameters)
	inferParameterString := string(inferParameterJSON)

	// Container VolumeMounts parameters
	inferCodeConPath := codePrefix
	inferModelConPath := dataPrefix + inferModelParent

	// Env parameters for edge
	inferModelURL := dataPrefix + inferModelPath

	// Configure container mounting and Env information by initial ContainerPara
	var inferContainer *ContainerPara = new(ContainerPara)
	inferContainer.volumeMountList = []string{inferCodeConPath, inferModelConPath}
	inferContainer.volumeList = []string{inferCodePath, inferModelParent}
	inferContainer.volumeMapName = []string{"code", "model"}
	inferContainer.env = map[string]string{
		"WORKER_NAME":           "inferworker-" + utilrand.String(5),
		"PARAMETERS":            inferParameterString,
		"MODEL_URL":             inferModelURL,
		"NAMESPACE":             job.Namespace,
		"HARD_SAMPLE_ALGORITHM": job.Spec.DeploySpec.HardExampleMining.Name,
		"LC_SERVER":             jc.cfg.LC.Server,
	}

	// create edge pod
	err = jc.generatePod(job, inferWorkerSpec, "inference", inferContainer)
	return err
}

// generatePod forms a pod for train and eval for incrementaljob
func (jc *IncrementalJobController) generatePod(job *sednav1.IncrementalLearningJob, workerSpec sednav1.CommonWorkerSpec, workerType string, containerPara *ContainerPara) error {
	var volumeMounts []v1.VolumeMount
	var volumes []v1.Volume
	var envs []v1.EnvVar
	var nodeName string
	if workerType == "inference" {
		nodeName = job.Spec.DeploySpec.NodeName
	} else {
		nodeName = job.Spec.NodeName
	}
	ctx := context.Background()
	// get baseImgURL from imageHub based on user's configuration in job CRD
	frameName := workerSpec.FrameworkType
	frameVersion := workerSpec.FrameworkVersion
	baseImgURL, err := MatchContainerBaseImage(jc.cfg.ImageHub, frameName, frameVersion)
	// TODO: if matched image is empty, the pod creation process will not proceed, return error directly.
	if err != nil {
		klog.Warningf("incrementallearning job %v/%v %v worker matching container base image occurs error:%v", job.Namespace, job.Name, workerType, err)
		return err
	}
	volumeMounts, volumes = CreateVolumeMap(containerPara)
	envs = CreateEnvVars(containerPara.env)
	podSpec := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: job.Namespace,
			Name:      jc.generatePodName(job.Name, workerType),
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(job, sednav1.SchemeGroupVersion.WithKind("IncrementalLearningJob")),
			},
			Labels: GenerateLabels(job),
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			HostNetwork:   true,
			NodeName:      nodeName,
			Containers: []v1.Container{
				{Name: "container-" + job.Name + "-" + strings.ToLower(workerType) + "-" + utilrand.String(5),
					Image:        baseImgURL,
					Args:         []string{workerSpec.ScriptBootFile},
					Env:          envs,
					VolumeMounts: volumeMounts,
				}},
			Volumes: volumes,
		},
	}
	pod, err := jc.kubeClient.CoreV1().Pods(job.Namespace).Create(ctx, podSpec, metav1.CreateOptions{})
	if err != nil {
		klog.Warningf("failed to create %s pod %s for incrementallearning job %v/%v, err:%s", workerType, pod.Name, job.Namespace, job.Name, err)
		return err
	}
	klog.V(2).Infof("%s pod %s is created successfully for incrementallearning job %v/%v", workerType, pod.Name, job.Namespace, job.Name)
	return nil
}

// GetName returns the name of the incrementallearning job controller
func (jc *IncrementalJobController) GetName() string {
	return "IncrementalLearningJobController"
}

// NewIncrementalJobController creates a new IncrementalJob controller that keeps the relevant pods
// in sync with their corresponding IncrementalJob objects.
func NewIncrementalJobController(cfg *config.ControllerConfig) (FeatureControllerI, error) {
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
	jobInformer := jobInformerFactory.Sedna().V1alpha1().IncrementalLearningJobs()

	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: kubeClient.CoreV1().Events("")})

	jc := &IncrementalJobController{
		kubeClient: kubeClient,
		client:     crdclient.SednaV1alpha1(),
		podControl: k8scontroller.RealPodControl{
			KubeClient: kubeClient,
			Recorder:   eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "incrementallearningjob-controller"}),
		},

		queue:    workqueue.NewNamedRateLimitingQueue(workqueue.NewItemExponentialFailureRateLimiter(DefaultBackOff, MaxBackOff), "incrementallearningjob"),
		recorder: eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "incrementallearningjob-controller"}),
		cfg:      cfg,
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
