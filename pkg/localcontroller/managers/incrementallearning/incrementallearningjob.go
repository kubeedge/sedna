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
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path"
	"strconv"
	"strings"
	"sync"
	"time"

	"k8s.io/klog/v2"

	"github.com/kubeedge/sedna/cmd/sedna-lc/app/options"
	sednav1 "github.com/kubeedge/sedna/pkg/apis/sedna/v1alpha1"
	gmtypes "github.com/kubeedge/sedna/pkg/globalmanager/controllers/incrementallearning"
	"github.com/kubeedge/sedna/pkg/globalmanager/runtime"
	"github.com/kubeedge/sedna/pkg/localcontroller/db"
	clienttypes "github.com/kubeedge/sedna/pkg/localcontroller/gmclient"
	"github.com/kubeedge/sedna/pkg/localcontroller/managers/dataset"
	"github.com/kubeedge/sedna/pkg/localcontroller/managers/model"
	"github.com/kubeedge/sedna/pkg/localcontroller/storage"
	"github.com/kubeedge/sedna/pkg/localcontroller/trigger"
	"github.com/kubeedge/sedna/pkg/localcontroller/util"
	workertypes "github.com/kubeedge/sedna/pkg/localcontroller/worker"
)

// IncrementalLearningJob defines config for incremental-learning-job
type Job struct {
	sednav1.IncrementalLearningJob
	JobConfig *JobConfig
	Dataset   *dataset.Dataset
	Storage   storage.Storage
	Done      chan struct{}
}

// JobConfig defines config for incremental-learning-job
type JobConfig struct {
	UniqueIdentifier   string
	Rounds             int
	TrainTrigger       trigger.Base
	DeployTrigger      trigger.Base
	TriggerTime        time.Time
	TrainTriggerStatus string
	EvalTriggerStatus  string
	TrainDataURL       string
	EvalDataURL        string
	OutputDir          string
	OutputConfig       *OutputConfig
	DataSamples        *DataSamples
	TrainModel         *Model
	DeployModel        *Model
	EvalModels         []Model
	EvalResult         []Model
	Lock               sync.Mutex
}

type Model = clienttypes.Model

// OutputConfig defines config for job output
type OutputConfig struct {
	SamplesOutput map[string]string `json:"trainData"`
	TrainOutput   string            `json:"trainOutput"`
	EvalOutput    string            `json:"evalOutput"`
}

// DataSamples defines samples information
type DataSamples struct {
	Numbers            int
	TrainSamples       []string
	EvalVersionSamples [][]string
	EvalSamples        []string
}

// IncrementalLearningJob defines incremental-learning-job manager
type Manager struct {
	Client               clienttypes.ClientI
	WorkerMessageChannel chan workertypes.MessageContent
	DatasetManager       *dataset.Manager
	ModelManager         *model.Manager
	IncrementalJobMap    map[string]*Job
	VolumeMountPrefix    string
}

const (
	// JobIterationIntervalSeconds is interval time of each iteration of job
	JobIterationIntervalSeconds = 10
	// DatasetHandlerIntervalSeconds is interval time of handling dataset
	DatasetHandlerIntervalSeconds = 10
	// EvalSamplesCapacity is capacity of eval samples
	EvalSamplesCapacity = 5
	//KindName is kind of incremental-learning-job resource
	KindName = "incrementallearningjob"

	// TriggerReadyStatus is the ready status about trigger
	TriggerReadyStatus = "ready"
	// TriggerCompletedStatus is the completed status about trigger
	TriggerCompletedStatus = "completed"
)

// New creates a incremental-learning-job manager
func New(client clienttypes.ClientI, datasetManager *dataset.Manager,
	modelManager *model.Manager, options *options.LocalControllerOptions) *Manager {
	im := Manager{
		Client:               client,
		WorkerMessageChannel: make(chan workertypes.MessageContent, workertypes.MessageChannelCacheSize),
		DatasetManager:       datasetManager,
		ModelManager:         modelManager,
		IncrementalJobMap:    make(map[string]*Job),
		VolumeMountPrefix:    options.VolumeMountPrefix,
	}

	return &im
}

// Start starts incremental-learning-job manager
func (im *Manager) Start() error {
	go im.monitorWorker()

	return nil
}

// trainTask starts training task
func (im *Manager) trainTask(job *Job, currentRound int) error {
	jobConfig := job.JobConfig

	latestCond := im.getLatestCondition(job)
	jobStage := latestCond.Stage
	currentType := latestCond.Type

	if currentType == sednav1.ILJobStageCondWaiting {
		if job.Dataset == nil {
			return fmt.Errorf("job(name=%s) dataset not ready", jobConfig.UniqueIdentifier)
		}

		err := im.loadTrainModel(job)
		if err != nil {
			return fmt.Errorf("failed to sync train model, and waiting it: %v", err)
		}

		if currentRound < jobConfig.Rounds {
			currentRound = jobConfig.Rounds
			initTriggerStatus(jobConfig)
		}
	}

	if currentType == sednav1.ILJobStageCondWaiting && jobConfig.TrainTriggerStatus == TriggerReadyStatus {
		payload, ok, err := im.triggerTrainTask(job)
		if !ok {
			return nil
		}

		if err != nil {
			klog.Errorf("job(name=%s) complete the %sing phase triggering task failed, error: %v",
				jobConfig.UniqueIdentifier, jobStage, err)
			return err
		}

		err = im.Client.WriteMessage(payload, job.getHeader())
		if err != nil {
			klog.Errorf("job(name=%s) failed to write message: %v",
				jobConfig.UniqueIdentifier, err)
			return err
		}

		jobConfig.TrainTriggerStatus = TriggerCompletedStatus
		jobConfig.Rounds++
		forwardSamples(jobConfig, jobStage)
		klog.Infof("job(name=%s) complete the %sing phase triggering task successfully",
			jobConfig.UniqueIdentifier, jobStage)
	}

	return nil
}

// evalTask starts eval task
func (im *Manager) evalTask(job *Job) error {
	jobConfig := job.JobConfig

	latestCond := im.getLatestCondition(job)
	jobStage := latestCond.Stage
	currentType := latestCond.Type

	if currentType == sednav1.ILJobStageCondWaiting {
		err := im.loadDeployModel(job)
		if err != nil {
			return fmt.Errorf("failed to sync deploy model, and waiting it: %v", err)
		}
	}

	if currentType == sednav1.ILJobStageCondWaiting && jobConfig.EvalTriggerStatus == TriggerReadyStatus {
		payload, err := im.triggerEvalTask(job)
		if err != nil {
			klog.Errorf("job(name=%s) complete the %sing phase triggering task failed, error: %v",
				jobConfig.UniqueIdentifier, jobStage, err)
			return err
		}

		err = im.Client.WriteMessage(payload, job.getHeader())
		if err != nil {
			return err
		}

		jobConfig.EvalTriggerStatus = TriggerCompletedStatus
		forwardSamples(jobConfig, jobStage)
		klog.Infof("job(name=%s) complete the %sing phase triggering task successfully",
			jobConfig.UniqueIdentifier, jobStage)
	}

	return nil
}

// deployTask starts deploy task
func (im *Manager) deployTask(job *Job) {
	jobConfig := job.JobConfig
	var err error
	var neededDeploy bool

	neededDeploy, err = im.triggerDeployTask(job)
	status := clienttypes.UpstreamMessage{Phase: string(sednav1.ILJobDeploy)}

	if err == nil && neededDeploy {
		deployModel, err := im.deployModel(job)
		if err != nil {
			klog.Errorf("failed to deploy model for job(name=%s): %v", jobConfig.UniqueIdentifier, err)
		} else {
			klog.Infof("deployed model for job(name=%s) successfully", jobConfig.UniqueIdentifier)
		}
		if err != nil || deployModel == nil {
			status.Status = string(sednav1.ILJobStageCondFailed)
		} else {
			status.Status = string(sednav1.ILJobStageCondReady)
			status.Input = &clienttypes.Input{
				Models: []Model{
					*deployModel,
				},
			}
		}

		klog.Infof("job(name=%s) complete the %sing phase triggering task successfully",
			jobConfig.UniqueIdentifier, sednav1.ILJobDeploy)
	} else {
		// No need to deploy, just report completed status
		// TODO: instead of reporting deploy-completed, another more reasonable status
		klog.Infof("no need to deploy model for job(name=%s)", jobConfig.UniqueIdentifier)
		status.Status = string(sednav1.ILJobStageCondCompleted)
	}

	err = im.Client.WriteMessage(status, job.getHeader())
	if err != nil {
		klog.Errorf("job(name=%s) complete the %s task failed, error: %v",
			jobConfig.UniqueIdentifier, sednav1.ILJobDeploy, err)
	}

	klog.Infof("job(name=%s) complete the %s task successfully", jobConfig.UniqueIdentifier, sednav1.ILJobDeploy)
}

// startJob starts a job
func (im *Manager) startJob(name string) {
	var err error
	job := im.IncrementalJobMap[name]

	job.JobConfig = new(JobConfig)
	jobConfig := job.JobConfig
	jobConfig.UniqueIdentifier = name

	err = im.initJob(job)
	if err != nil {
		klog.Errorf("failed to init job (name=%s): %+v", jobConfig.UniqueIdentifier)
		return
	}

	klog.Infof("incremental job(name=%s) is started", name)
	defer klog.Infof("incremental learning job(name=%s) is stopped", name)

	cond := im.getLatestCondition(job)
	currentType := cond.Type
	jobStage := cond.Stage
	if jobStage == sednav1.ILJobTrain && currentType == sednav1.ILJobStageCondWaiting {
		go im.handleData(job)
	}

	currentRound := jobConfig.Rounds

	tick := time.NewTicker(JobIterationIntervalSeconds * time.Second)
	for {
		select {
		case <-job.Done:
			return
		default:
		}

		latestCond := im.getLatestCondition(job)
		jobStage := latestCond.Stage

		switch jobStage {
		case sednav1.ILJobTrain:
			err = im.trainTask(job, currentRound)
		case sednav1.ILJobEval:
			err = im.evalTask(job)
		default:
			klog.Errorf("invalid phase: %s", jobStage)
			continue
		}

		if err != nil {
			klog.Errorf("job(name=%s) complete the %s task failed, error: %v",
				jobConfig.UniqueIdentifier, jobStage, err)
		}

		<-tick.C
	}
}

// Insert inserts incremental-learning-job config to db
func (im *Manager) Insert(message *clienttypes.Message) error {
	name := util.GetUniqueIdentifier(message.Header.Namespace, message.Header.ResourceName, message.Header.ResourceKind)

	first := false
	job, ok := im.IncrementalJobMap[name]
	if !ok {
		job = &Job{}
		job.Storage = storage.Storage{IsLocalStorage: false}
		job.Done = make(chan struct{})
		im.IncrementalJobMap[name] = job
		first = true
	}

	if err := json.Unmarshal(message.Content, &job); err != nil {
		return err
	}

	credential := job.ObjectMeta.Annotations[runtime.SecretAnnotationKey]
	if credential != "" {
		if err := job.Storage.SetCredential(credential); err != nil {
			return fmt.Errorf("failed to set job(name=%s)'s storage credential, error: %+v", name, err)
		}
	}

	if first {
		go im.startJob(name)
	}

	if err := db.SaveResource(name, job.TypeMeta, job.ObjectMeta, job.Spec); err != nil {
		return err
	}

	return nil
}

// Delete deletes incremental-learning-job config in db
func (im *Manager) Delete(message *clienttypes.Message) error {
	name := util.GetUniqueIdentifier(message.Header.Namespace, message.Header.ResourceName, message.Header.ResourceKind)

	if job, ok := im.IncrementalJobMap[name]; ok && job.Done != nil {
		close(job.Done)
	}

	delete(im.IncrementalJobMap, name)

	if err := db.DeleteResource(name); err != nil {
		return err
	}

	return nil
}

// initJob inits the job object
func (im *Manager) initJob(job *Job) error {
	jobConfig := job.JobConfig
	jobConfig.Lock = sync.Mutex{}

	jobConfig.Rounds = 1
	initTriggerStatus(jobConfig)
	trainTrigger, err := newTrigger(job.Spec.TrainSpec.Trigger)
	if err != nil {
		return fmt.Errorf("failed to init train trigger: %+w", err)
	}
	deployTrigger, err := newTrigger(job.Spec.DeploySpec.Trigger)
	if err != nil {
		return fmt.Errorf("failed to init deploy trigger: %+w", err)
	}
	jobConfig.TrainTrigger = trainTrigger
	jobConfig.DeployTrigger = deployTrigger

	outputDir := job.Spec.OutputDir

	isLocalURL, err := job.Storage.IsLocalURL(outputDir)
	if err != nil {
		return fmt.Errorf("job(name=%s)'s output dir is invalid, error: %+v", job.Name, outputDir)
	}

	if isLocalURL {
		job.Storage.IsLocalStorage = true
		outputDir = util.AddPrefixPath(im.VolumeMountPrefix, outputDir)
	}

	jobConfig.OutputDir = outputDir

	if err := job.createOutputDir(jobConfig); err != nil {
		return err
	}

	return nil
}

func initTriggerStatus(jobConfig *JobConfig) {
	jobConfig.TrainTriggerStatus = TriggerReadyStatus
	jobConfig.EvalTriggerStatus = TriggerReadyStatus
}

func newTrigger(t sednav1.Trigger) (trigger.Base, error) {
	// convert trigger to map
	triggerMap := make(map[string]interface{})
	c, err := json.Marshal(t)
	if err != nil {
		return nil, err
	}

	err = json.Unmarshal(c, &triggerMap)
	if err != nil {
		return nil, err
	}
	return trigger.NewTrigger(triggerMap)
}

// getTrainOrEvalModel gets train model or eval model from job conditions
func (im *Manager) getTrainOrEvalModel(job *Job, jobStage sednav1.ILJobStage) *Model {
	jobConditions := job.Status.Conditions

	// TODO: runtime.type changes to common.type for gm and lc
	var models []runtime.Model

	for i := len(jobConditions) - 1; i >= 0; i-- {
		var cond gmtypes.IncrementalCondData
		jobCond := jobConditions[i]
		if jobCond.Stage == sednav1.ILJobTrain && jobCond.Type == sednav1.ILJobStageCondCompleted {
			if err := (&cond).Unmarshal([]byte(jobCond.Data)); err != nil {
				continue
			}

			if cond.Output == nil {
				continue
			}
			// models list has two model, first is deploy model, second is trained model
			models = cond.Output.Models

			break
		}
	}

	// models must have two model file info which are output of train,
	// first model will be used for inference if it evaluated as excellent, second model will be used for retaining.
	if len(models) != 2 {
		return nil
	}

	switch jobStage {
	case sednav1.ILJobTrain:
		return &Model{Format: models[1].Format, URL: models[1].URL}
	case sednav1.ILJobEval:
		return &Model{Format: models[0].Format, URL: models[0].URL}
	}

	return nil
}

// triggerTrainTask triggers the train task
func (im *Manager) triggerTrainTask(job *Job) (interface{}, bool, error) {
	var err error
	jobConfig := job.JobConfig

	const numOfSamples = "num_of_samples"
	samples := map[string]interface{}{
		numOfSamples: len(jobConfig.DataSamples.TrainSamples),
	}

	isTrigger := jobConfig.TrainTrigger.Trigger(samples)

	if !isTrigger {
		return nil, false, nil
	}

	var m *Model

	latestCondition := im.getLatestCondition(job)
	rounds := jobConfig.Rounds
	if rounds <= 1 {
		m = jobConfig.TrainModel
	} else {
		m = im.getTrainOrEvalModel(job, latestCondition.Stage)
		if m == nil {
			return nil, false, err
		}
	}

	var dataIndexURL string
	jobConfig.TrainDataURL, dataIndexURL, err = im.writeSamples(job, jobConfig.DataSamples.TrainSamples,
		jobConfig.OutputConfig.SamplesOutput["train"], rounds, job.Dataset.Spec.Format, job.Dataset.URLPrefix)
	if err != nil {
		klog.Errorf("job(name=%s) train phase: write samples to the file(%s) is failed, error: %v",
			jobConfig.UniqueIdentifier, jobConfig.TrainDataURL, err)
		return nil, false, err
	}

	dataURL := jobConfig.TrainDataURL
	outputDir := strings.Join([]string{jobConfig.OutputConfig.TrainOutput, strconv.Itoa(rounds)}, "/")
	if job.Storage.IsLocalStorage {
		dataURL = util.TrimPrefixPath(im.VolumeMountPrefix, dataURL)
		dataIndexURL = util.TrimPrefixPath(im.VolumeMountPrefix, dataIndexURL)
		outputDir = util.TrimPrefixPath(im.VolumeMountPrefix, outputDir)
	}

	input := clienttypes.Input{
		Models:       []Model{*m},
		DataURL:      dataURL,
		DataIndexURL: dataIndexURL,
		OutputDir:    outputDir,
	}
	msg := clienttypes.UpstreamMessage{
		Phase:  string(sednav1.ILJobTrain),
		Status: string(sednav1.ILJobStageCondReady),
		Input:  &input,
	}
	jobConfig.TriggerTime = time.Now()
	return &msg, true, nil
}

// triggerEvalTask triggers the eval task
func (im *Manager) triggerEvalTask(job *Job) (*clienttypes.UpstreamMessage, error) {
	jobConfig := job.JobConfig
	var err error

	latestCondition := im.getLatestCondition(job)

	m := im.getTrainOrEvalModel(job, latestCondition.Stage)
	if m == nil {
		return nil, err
	}

	models := []Model{*m, {
		Format: jobConfig.DeployModel.Format,
		URL:    jobConfig.DeployModel.URL,
	}}
	// EvalModels has two models, first is trained model, second is deployed model
	jobConfig.EvalModels = models

	var dataIndexURL string
	rounds := jobConfig.Rounds
	jobConfig.EvalDataURL, dataIndexURL, err = im.writeSamples(job, jobConfig.DataSamples.EvalSamples, jobConfig.OutputConfig.SamplesOutput["eval"],
		rounds, job.Dataset.Spec.Format, job.Dataset.URLPrefix)
	if err != nil {
		klog.Errorf("job(name=%s) eval phase: write samples to the file(%s) is failed, error: %v",
			jobConfig.UniqueIdentifier, jobConfig.EvalDataURL, err)
		return nil, err
	}

	dataURL := jobConfig.EvalDataURL
	if job.Storage.IsLocalStorage {
		dataURL = util.TrimPrefixPath(im.VolumeMountPrefix, dataURL)
		dataIndexURL = util.TrimPrefixPath(im.VolumeMountPrefix, dataIndexURL)
	}

	input := clienttypes.Input{
		Models:       models,
		DataURL:      dataURL,
		DataIndexURL: dataIndexURL,
	}
	msg := &clienttypes.UpstreamMessage{
		Phase:  string(sednav1.ILJobEval),
		Status: string(sednav1.ILJobStageCondReady),
		Input:  &input,
	}

	return msg, nil
}

// triggerDeployTask triggers the deploy task
func (im *Manager) triggerDeployTask(job *Job) (bool, error) {
	jobConfig := job.JobConfig

	// EvalResult must has two models info, first is trained model, second is deployed model.
	if len(jobConfig.EvalResult) != 2 {
		return false, fmt.Errorf("expected 2 evaluation results, actual: %d", len(jobConfig.EvalResult))
	}

	getMetrics := func(metrics map[string]interface{}) (map[string][]float64, error) {
		var err error
		bytes, err := json.Marshal(metrics)
		if err != nil {
			return nil, err
		}

		data := make(map[string][]float64)
		if err := json.Unmarshal(bytes, &data); err != nil {
			return nil, err
		}
		return data, err
	}

	newMetrics, err := getMetrics(jobConfig.EvalResult[0].Metrics)
	if err != nil {
		return false, err
	}
	oldMetrics, err := getMetrics(jobConfig.EvalResult[1].Metrics)
	if err != nil {
		return false, err
	}
	metricDelta := make(map[string]interface{})

	for metric := range newMetrics {
		// keep the full metrics
		metricDelta[metric] = newMetrics[metric]
		var l []float64
		for i := range newMetrics[metric] {
			l = append(l, newMetrics[metric][i]-oldMetrics[metric][i])
		}
		metricDelta[metric+"_delta"] = l
	}
	tt := job.Spec.DeploySpec.Trigger

	// convert tt to map
	triggerMap := make(map[string]interface{})
	c, err := json.Marshal(tt)
	if err != nil {
		return false, err
	}

	err = json.Unmarshal(c, &triggerMap)
	if err != nil {
		return false, err
	}

	return jobConfig.DeployTrigger.Trigger(metricDelta), nil
}

// deployModel deploys model
func (im *Manager) deployModel(job *Job) (*Model, error) {
	jobConfig := job.JobConfig

	trainedModel := jobConfig.EvalModels[0].URL
	deployModel := jobConfig.EvalModels[1].URL
	if job.Storage.IsLocalStorage {
		trainedModel = util.AddPrefixPath(im.VolumeMountPrefix, trainedModel)
	}

	if err := job.updateDeployModel(deployModel, trainedModel); err != nil {
		return nil, err
	}

	klog.Infof("job(name=%s) deploys model(url=%s) successfully", jobConfig.UniqueIdentifier, trainedModel)

	return &jobConfig.EvalModels[0], nil
}

func (job *Job) updateDeployModel(deployModel string, newModel string) error {
	if err := job.Storage.CopyFile(newModel, deployModel); err != nil {
		return fmt.Errorf("copy model(url=%s) to the deploy model(url=%s) failed, error: %+v",
			newModel, deployModel, err)
	}

	klog.Infof("copy model(url=%s) to the deploy model(url=%s) successfully", newModel, deployModel)
	return nil
}

// createOutputDir creates the job output dir
func (job *Job) createOutputDir(jobConfig *JobConfig) error {
	outputDir := jobConfig.OutputDir

	dirNames := []string{"data/train", "data/eval", "train", "eval"}

	if job.Storage.IsLocalStorage {
		if err := util.CreateFolder(outputDir); err != nil {
			klog.Errorf("job(name=%s) create fold %s failed", jobConfig.UniqueIdentifier, outputDir)
			return err
		}

		for _, v := range dirNames {
			dir := path.Join(outputDir, v)
			if err := util.CreateFolder(dir); err != nil {
				klog.Errorf("job(name=%s) create fold %s failed", jobConfig.UniqueIdentifier, dir)
				return err
			}
		}
	}

	outputConfig := OutputConfig{
		SamplesOutput: map[string]string{
			"train": strings.Join([]string{strings.TrimRight(outputDir, "/"), dirNames[0]}, "/"),
			"eval":  strings.Join([]string{strings.TrimRight(outputDir, "/"), dirNames[1]}, "/"),
		},
		TrainOutput: strings.Join([]string{strings.TrimRight(outputDir, "/"), dirNames[2]}, "/"),
		EvalOutput:  strings.Join([]string{strings.TrimRight(outputDir, "/"), dirNames[3]}, "/"),
	}
	jobConfig.OutputConfig = &outputConfig

	return nil
}

func (im *Manager) getLatestCondition(job *Job) sednav1.ILJobCondition {
	jobConditions := job.Status.Conditions
	var latestCondition sednav1.ILJobCondition = sednav1.ILJobCondition{}
	if len(jobConditions) > 0 {
		// get latest pod and pod status
		latestCondition = jobConditions[len(jobConditions)-1]
	}
	return latestCondition
}

func (im *Manager) getModel(namespace string, name string) (sednav1.Model, error) {
	modelName := util.GetUniqueIdentifier(namespace, name, model.KindName)
	model, ok := im.ModelManager.GetModel(modelName)
	if !ok {
		return model, fmt.Errorf("not exists model(name=%s)", modelName)
	}
	return model, nil
}

// loadTrainModel loads initial model information for training.
func (im *Manager) loadTrainModel(job *Job) error {
	jobConfig := job.JobConfig

	if jobConfig.TrainModel == nil {
		initialModel, err := im.getModel(job.Namespace, job.Spec.InitialModel.Name)
		if err != nil {
			return err
		}

		jobConfig.TrainModel = new(Model)
		format := initialModel.Spec.Format
		url := initialModel.Spec.URL
		jobConfig.TrainModel.Format = format
		jobConfig.TrainModel.URL = url
	}
	return nil
}

// loadDeployModel loads model information for deploying.
func (im *Manager) loadDeployModel(job *Job) error {
	jobConfig := job.JobConfig

	if jobConfig.DeployModel == nil {
		evalModel, err := im.getModel(job.Namespace, job.Spec.DeploySpec.Model.Name)
		if err != nil {
			return err
		}

		jobConfig.DeployModel = new(Model)
		jobConfig.DeployModel.Format = evalModel.Spec.Format
		jobConfig.DeployModel.URL = evalModel.Spec.URL
	}
	return nil
}

// loadDataset loads dataset information
func (im *Manager) loadDataset(job *Job) error {
	if job.Dataset != nil {
		// already loaded
		return nil
	}

	datasetName := util.GetUniqueIdentifier(job.Namespace, job.Spec.Dataset.Name, dataset.KindName)
	dataset, ok := im.DatasetManager.GetDataset(datasetName)
	if !ok || dataset == nil {
		return fmt.Errorf("not exists dataset(name=%s)", datasetName)
	}

	jobConfig := job.JobConfig
	jobConfig.DataSamples = &DataSamples{
		Numbers:            0,
		TrainSamples:       make([]string, 0),
		EvalVersionSamples: make([][]string, 0),
		EvalSamples:        make([]string, 0),
	}

	job.Dataset = dataset
	return nil
}

// handleData updates samples information
func (im *Manager) handleData(job *Job) {
	tick := time.NewTicker(DatasetHandlerIntervalSeconds * time.Second)

	jobConfig := job.JobConfig
	iterCount := 0
	for {
		select {
		case <-job.Done:
			return
		default:
		}

		// in case dataset is not synced to LC before job synced to LC
		// here call loadDataset in each period
		err := im.loadDataset(job)
		if iterCount%100 == 0 {
			klog.Infof("job(name=%s) handling dataset", jobConfig.UniqueIdentifier)
		}
		iterCount++
		if err != nil {
			klog.Warningf("job(name=%s) failed to load dataset, and waiting it: %v",
				jobConfig.UniqueIdentifier,
				err)
			<-tick.C
			continue
		}

		dataset := job.Dataset

		if dataset.DataSource != nil && len(dataset.DataSource.TrainSamples) > jobConfig.DataSamples.Numbers {
			samples := dataset.DataSource.TrainSamples
			trainNum := int(job.Spec.Dataset.TrainProb * float64(len(samples)-jobConfig.DataSamples.Numbers))

			jobConfig.Lock.Lock()
			jobConfig.DataSamples.TrainSamples = append(jobConfig.DataSamples.TrainSamples,
				samples[(jobConfig.DataSamples.Numbers+1):(jobConfig.DataSamples.Numbers+trainNum+1)]...)
			klog.Infof("job(name=%s) current train samples nums is %d",
				jobConfig.UniqueIdentifier, len(jobConfig.DataSamples.TrainSamples))

			jobConfig.DataSamples.EvalVersionSamples = append(jobConfig.DataSamples.EvalVersionSamples,
				samples[(jobConfig.DataSamples.Numbers+trainNum+1):])
			jobConfig.Lock.Unlock()

			for _, v := range jobConfig.DataSamples.EvalVersionSamples {
				jobConfig.DataSamples.EvalSamples = append(jobConfig.DataSamples.EvalSamples, v...)
			}
			klog.Infof("job(name=%s) current eval samples nums is %d",
				jobConfig.UniqueIdentifier, len(jobConfig.DataSamples.EvalSamples))

			jobConfig.DataSamples.Numbers = len(samples)
		}

		<-tick.C
	}
}

// createFile creates data file and data index file
func createFile(dir string, format string, isLocalStorage bool) (string, string) {
	switch format {
	case dataset.TXTFormat:
		if isLocalStorage {
			return path.Join(dir, "data.txt"), ""
		}
		return strings.Join([]string{dir, "data.txt"}, "/"), strings.Join([]string{dir, "dataIndex.txt"}, "/")
	}
	return "", ""
}

// writeSamples writes samples information to a file
func (im *Manager) writeSamples(job *Job, samples []string, dir string, rounds int, format string, urlPrefix string) (string, string, error) {
	subDir := strings.Join([]string{dir, strconv.Itoa(rounds)}, "/")
	fileURL, absURLFile := createFile(subDir, format, job.Dataset.Storage.IsLocalStorage)

	if job.Storage.IsLocalStorage {
		if err := util.CreateFolder(subDir); err != nil {
			return "", "", err
		}
		if err := im.writeByLine(samples, fileURL); err != nil {
			return "", "", err
		}

		if !job.Dataset.Storage.IsLocalStorage {
			tempSamples := util.ParsingDatasetIndex(samples, urlPrefix)
			if err := im.writeByLine(tempSamples, absURLFile); err != nil {
				return "", "", err
			}
		}

		return fileURL, absURLFile, nil
	}

	temporaryDir, err := util.CreateTemporaryDir()
	if err != nil {
		return "", "", err
	}

	localFileURL, localAbsURLFile := createFile(temporaryDir, format, job.Dataset.Storage.IsLocalStorage)

	if err := im.writeByLine(samples, localFileURL); err != nil {
		return "", "", err
	}

	if err := job.Storage.Upload(localFileURL, fileURL); err != nil {
		return "", "", err
	}

	tempSamples := util.ParsingDatasetIndex(samples, urlPrefix)

	if err := im.writeByLine(tempSamples, localAbsURLFile); err != nil {
		return "", "", err
	}

	if err := job.Storage.Upload(localAbsURLFile, absURLFile); err != nil {
		return "", "", err
	}

	defer os.RemoveAll(localFileURL)
	defer os.RemoveAll(localAbsURLFile)

	return fileURL, absURLFile, nil
}

// writeByLine writes file by line
func (im *Manager) writeByLine(samples []string, fileURL string) error {
	file, err := os.Create(fileURL)
	if err != nil {
		klog.Errorf("create file(%s) failed", fileURL)
		return err
	}

	w := bufio.NewWriter(file)
	for _, line := range samples {
		_, _ = fmt.Fprintln(w, line)
	}
	if err := w.Flush(); err != nil {
		klog.Errorf("write file(%s) failed", fileURL)
		return err
	}

	if err := file.Close(); err != nil {
		klog.Errorf("close file failed, error: %v", err)
		return err
	}

	return nil
}

// monitorWorker monitors message from worker
func (im *Manager) monitorWorker() {
	for {
		workerMessageChannel := im.WorkerMessageChannel
		workerMessage, ok := <-workerMessageChannel
		if !ok {
			break
		}
		klog.V(4).Infof("handling worker message %+v", workerMessage)

		name := util.GetUniqueIdentifier(workerMessage.Namespace, workerMessage.OwnerName, workerMessage.OwnerKind)

		job, ok := im.IncrementalJobMap[name]
		if !ok {
			continue
		}

		// TODO: filter some worker messages out
		wo := clienttypes.Output{}
		wo.Models = workerMessage.Results
		wo.OwnerInfo = workerMessage.OwnerInfo

		msg := &clienttypes.UpstreamMessage{
			Phase:  workerMessage.Kind,
			Status: workerMessage.Status,
			Output: &wo,
		}

		if err := im.Client.WriteMessage(msg, job.getHeader()); err != nil {
			klog.Errorf("job(name=%s) failed to write message: %v", name, err)
			continue
		}

		im.handleWorkerMessage(job, workerMessage)
	}
}

// handleWorkerMessage handles message from worker
func (im *Manager) handleWorkerMessage(job *Job, workerMessage workertypes.MessageContent) {
	latestCond := im.getLatestCondition(job)
	jobStage := strings.ToLower(string(latestCond.Stage))
	workerKind := strings.ToLower(workerMessage.Kind)

	if jobStage != workerKind {
		klog.Warningf("job(name=%s) %s phase get worker(kind=%s)", job.JobConfig.UniqueIdentifier,
			jobStage, workerKind)
		return
	}

	var models []Model
	for _, result := range workerMessage.Results {
		metrics := make(map[string]interface{})
		if m, ok := result["metrics"]; ok {
			bytes, err := json.Marshal(m)
			if err != nil {
				return
			}

			err = json.Unmarshal(bytes, &metrics)
			if err != nil {
				klog.Warningf("failed to unmarshal the worker(name=%s) metrics %v, err: %v",
					workerMessage.Name,
					m,
					err)
			}
		}

		model := Model{
			Format:  result["format"].(string),
			URL:     result["url"].(string),
			Metrics: metrics}
		models = append(models, model)
	}

	workerStatus := workerMessage.Status
	jobName := job.JobConfig.UniqueIdentifier

	if workerStatus == workertypes.CompletedStatus {
		klog.Infof("job(name=%s) complete the %s task successfully", jobName, jobStage)
		switch latestCond.Stage {
		case sednav1.ILJobEval:
			job.JobConfig.EvalResult = models
			// when eval worker is complete, the deploy task starts immediately without waiting for the notification of GM.
			im.deployTask(job)
		}
	}
}

// forwardSamples deletes the samples information in the memory
func forwardSamples(jobConfig *JobConfig, jobStage sednav1.ILJobStage) {
	switch jobStage {
	case sednav1.ILJobTrain:
		jobConfig.Lock.Lock()
		jobConfig.DataSamples.TrainSamples = jobConfig.DataSamples.TrainSamples[:0]
		jobConfig.Lock.Unlock()
	case sednav1.ILJobEval:
		if len(jobConfig.DataSamples.EvalVersionSamples) > EvalSamplesCapacity {
			jobConfig.DataSamples.EvalVersionSamples = jobConfig.DataSamples.EvalVersionSamples[1:]
		}
	}
}

// AddWorkerMessage adds worker messages
func (im *Manager) AddWorkerMessage(message workertypes.MessageContent) {
	im.WorkerMessageChannel <- message
}

// GetName returns name of the manager
func (im *Manager) GetName() string {
	return KindName
}

func (job *Job) getHeader() clienttypes.MessageHeader {
	return clienttypes.MessageHeader{
		Namespace:    job.Namespace,
		ResourceKind: job.Kind,
		ResourceName: job.Name,
		Operation:    clienttypes.StatusOperation,
	}
}
