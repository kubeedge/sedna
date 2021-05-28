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

package manager

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
	"github.com/kubeedge/sedna/pkg/localcontroller/db"
	"github.com/kubeedge/sedna/pkg/localcontroller/gmclient"
	"github.com/kubeedge/sedna/pkg/localcontroller/storage"
	"github.com/kubeedge/sedna/pkg/localcontroller/trigger"
	"github.com/kubeedge/sedna/pkg/localcontroller/util"
)

const (
	//LifelongLearningJobKind is kind of lifelong-learning-job resource
	LifelongLearningJobKind = "lifelonglearningjob"
)

// LifelongLearningJobManager defines lifelong-learning-job Manager
type LifelongLearningJobManager struct {
	Client                 gmclient.ClientI
	WorkerMessageChannel   chan WorkerMessage
	DatasetManager         *DatasetManager
	LifelongLearningJobMap map[string]*LifelongLearningJob
	VolumeMountPrefix      string
}

// LifelongLearningJob defines config for lifelong-learning-job
type LifelongLearningJob struct {
	sednav1.LifelongLearningJob
	Dataset   *Dataset
	Done      chan struct{}
	Storage   storage.Storage
	JobConfig *LLJobConfig
}

// LLJobConfig defines config for lifelong-learning-job
type LLJobConfig struct {
	UniqueIdentifier string
	Version          int
	Phase            string
	WorkerStatus     string
	TrainTrigger     trigger.Base
	TriggerStatus    string
	TriggerTime      time.Time
	TrainDataURL     string
	EvalDataURL      string
	OutputDir        string
	OutputConfig     *LLOutputConfig
	DataSamples      *LLDataSamples
	TrainModel       *ModelInfo
	DeployModel      *ModelInfo
	EvalResult       *ModelInfo
	Lock             sync.Mutex
}

// LLOutputConfig defines config for job output
type LLOutputConfig struct {
	SamplesOutput map[string]string
	TrainOutput   string
	EvalOutput    string
}

// LLDataSamples defines samples information
type LLDataSamples struct {
	Numbers            int
	TrainSamples       []string
	EvalVersionSamples [][]string
	EvalSamples        []string
}

const (
	// LLJobIterationIntervalSeconds is interval time of each iteration of job
	LLJobIterationIntervalSeconds = 10
	// LLHandlerDataIntervalSeconds is interval time of handling dataset
	LLHandlerDataIntervalSeconds = 10
	// LLLLEvalSamplesCapacity is capacity of eval samples
	LLEvalSamplesCapacity = 5
)

// NewLifelongLearningJobManager creates a lifelong-learning-job manager
func NewLifelongLearningJobManager(client gmclient.ClientI, datasetManager *DatasetManager,
	modelManager *ModelManager, options *options.LocalControllerOptions) *LifelongLearningJobManager {
	lm := LifelongLearningJobManager{
		Client:                 client,
		WorkerMessageChannel:   make(chan WorkerMessage, WorkerMessageChannelCacheSize),
		DatasetManager:         datasetManager,
		LifelongLearningJobMap: make(map[string]*LifelongLearningJob),
		VolumeMountPrefix:      options.VolumeMountPrefix,
	}

	return &lm
}

// Insert inserts lifelong-learning-job config to db
func (lm *LifelongLearningJobManager) Insert(message *gmclient.Message) error {
	name := util.GetUniqueIdentifier(message.Header.Namespace, message.Header.ResourceName, message.Header.ResourceKind)

	first := false
	job, ok := lm.LifelongLearningJobMap[name]
	if !ok {
		job = &LifelongLearningJob{}
		job.Storage = storage.Storage{IsLocalStorage: false}
		job.Done = make(chan struct{})
		lm.LifelongLearningJobMap[name] = job
		first = true
	}

	if err := json.Unmarshal(message.Content, &job); err != nil {
		return err
	}

	credential := job.ObjectMeta.Annotations[CredentialAnnotationKey]
	if credential != "" {
		if err := job.Storage.SetCredential(credential); err != nil {
			return fmt.Errorf("failed to set job(name=%s)'s storage credential, error: %+v", name, err)
		}
	}

	if first {
		go lm.startJob(name)
	}

	if err := db.SaveResource(name, job.TypeMeta, job.ObjectMeta, job.Spec); err != nil {
		return err
	}

	return nil
}

// startJob starts a job
func (lm *LifelongLearningJobManager) startJob(name string) {
	var err error
	job, ok := lm.LifelongLearningJobMap[name]
	if !ok {
		return
	}

	job.JobConfig = new(LLJobConfig)
	jobConfig := job.JobConfig
	jobConfig.UniqueIdentifier = name

	err = lm.initJob(job)
	if err != nil {
		klog.Errorf("failed to init job (name=%s): %+v", jobConfig.UniqueIdentifier)
		return
	}

	klog.Infof("lifelong learning job(name=%s) is started", name)
	defer klog.Infof("lifelong learning job(name=%s) is stopped", name)
	go lm.handleData(job)

	tick := time.NewTicker(LLJobIterationIntervalSeconds * time.Second)
	for {
		select {
		case <-job.Done:
			return
		default:
		}

		if job.Dataset == nil {
			klog.V(3).Infof("job(name=%s) dataset not ready",
				jobConfig.UniqueIdentifier)

			<-tick.C
			continue
		}

		switch jobConfig.Phase {
		case TrainPhase:
			err = lm.trainTask(job)
		case EvalPhase:
			err = lm.evalTask(job)
		case DeployPhase:
			err = lm.deployTask(job)
		default:
			klog.Errorf("invalid phase: %s", jobConfig.Phase)
			continue
		}

		if err != nil {
			klog.Errorf("job(name=%s) complete the %s task failed, error: %v",
				jobConfig.UniqueIdentifier, jobConfig.Phase, err)
		}

		<-tick.C
	}
}

// trainTask starts training task
func (lm *LifelongLearningJobManager) trainTask(job *LifelongLearningJob) error {
	jobConfig := job.JobConfig

	if jobConfig.WorkerStatus == WorkerReadyStatus && jobConfig.TriggerStatus == TriggerReadyStatus {
		payload, ok, err := lm.triggerTrainTask(job)
		if !ok {
			return nil
		}

		if err != nil {
			klog.Errorf("job(name=%s) complete the %sing phase triggering task failed, error: %v",
				jobConfig.UniqueIdentifier, jobConfig.Phase, err)
			return err
		}

		err = lm.Client.WriteMessage(payload, job.getHeader())
		if err != nil {
			klog.Errorf("job(name=%s) failed to write message: %v",
				jobConfig.UniqueIdentifier, err)
			return err
		}

		jobConfig.TriggerStatus = TriggerCompletedStatus

		klog.Infof("job(name=%s) complete the %sing phase triggering task successfully",
			jobConfig.UniqueIdentifier, jobConfig.Phase)
	}

	if jobConfig.WorkerStatus == WorkerFailedStatus {
		klog.Warningf("found the %sing phase worker that ran failed, "+
			"back the training phase triggering task", jobConfig.Phase)
		backLLTaskStatus(jobConfig)
	}

	if jobConfig.WorkerStatus == WorkerCompletedStatus {
		klog.Infof("job(name=%s) complete the %s task successfully", jobConfig.UniqueIdentifier, jobConfig.Phase)
		nextLLTask(jobConfig)
	}

	return nil
}

// evalTask starts eval task
func (lm *LifelongLearningJobManager) evalTask(job *LifelongLearningJob) error {
	jobConfig := job.JobConfig

	if jobConfig.WorkerStatus == WorkerReadyStatus && jobConfig.TriggerStatus == TriggerReadyStatus {
		payload, err := lm.triggerEvalTask(job)
		if err != nil {
			klog.Errorf("job(name=%s) complete the %sing phase triggering task failed, error: %v",
				jobConfig.UniqueIdentifier, jobConfig.Phase, err)
			return err
		}

		err = lm.Client.WriteMessage(payload, job.getHeader())
		if err != nil {
			return err
		}

		jobConfig.TriggerStatus = TriggerCompletedStatus

		klog.Infof("job(name=%s) complete the %sing phase triggering task successfully",
			jobConfig.UniqueIdentifier, jobConfig.Phase)
	}

	if jobConfig.WorkerStatus == WorkerFailedStatus {
		msg := fmt.Sprintf("job(name=%s) found the %sing phase worker that ran failed, "+
			"back the training phase triggering task", jobConfig.UniqueIdentifier, jobConfig.Phase)
		klog.Errorf(msg)
		return fmt.Errorf(msg)
	}

	if jobConfig.WorkerStatus == WorkerCompletedStatus {
		klog.Infof("job(name=%s) complete the %s task successfully", jobConfig.UniqueIdentifier, jobConfig.Phase)
		nextLLTask(jobConfig)
	}

	return nil
}

// deployTask starts deploy task
func (lm *LifelongLearningJobManager) deployTask(job *LifelongLearningJob) error {
	jobConfig := job.JobConfig

	if jobConfig.WorkerStatus == WorkerReadyStatus && jobConfig.TriggerStatus == TriggerReadyStatus {
		status := UpstreamMessage{}
		status.Phase = DeployPhase
		deployModel, err := lm.deployModel(job)
		if err != nil {
			klog.Errorf("failed to deploy model for job(name=%s): %v", jobConfig.UniqueIdentifier, err)
		} else {
			klog.Infof("deployed model for job(name=%s) successfully", jobConfig.UniqueIdentifier)
		}
		if err != nil || deployModel == nil {
			status.Status = WorkerFailedStatus
		} else {
			status.Status = WorkerReadyStatus
			status.Input = &WorkerInput{
				Models: []ModelInfo{
					*deployModel,
				},
			}
		}

		if err = lm.Client.WriteMessage(status, job.getHeader()); err != nil {
			return err
		}

		jobConfig.TriggerStatus = TriggerCompletedStatus
	}

	nextLLTask(jobConfig)

	klog.Infof("job(name=%s) complete the deploy task successfully", jobConfig.UniqueIdentifier)

	return nil
}

// triggerTrainTask triggers the train task
func (lm *LifelongLearningJobManager) triggerTrainTask(job *LifelongLearningJob) (interface{}, bool, error) {
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

	jobConfig.Version++

	var dataIndexURL string
	jobConfig.TrainDataURL, dataIndexURL, err = job.writeLLJSamples(jobConfig.DataSamples.TrainSamples,
		jobConfig.OutputConfig.SamplesOutput["train"])
	if err != nil {
		klog.Errorf("train phase: write samples to the file(%s) is failed, error: %v", jobConfig.TrainDataURL, err)
		return nil, false, err
	}

	dataURL := jobConfig.TrainDataURL
	outputDir := strings.Join([]string{jobConfig.OutputConfig.TrainOutput, strconv.Itoa(jobConfig.Version)}, "/")
	if job.Storage.IsLocalStorage {
		dataURL = util.TrimPrefixPath(lm.VolumeMountPrefix, dataURL)
		dataIndexURL = util.TrimPrefixPath(lm.VolumeMountPrefix, dataIndexURL)
		outputDir = util.TrimPrefixPath(lm.VolumeMountPrefix, outputDir)
	}

	input := WorkerInput{
		DataURL:      dataURL,
		DataIndexURL: dataIndexURL,
		OutputDir:    outputDir,
	}
	msg := UpstreamMessage{
		Phase:  TrainPhase,
		Status: WorkerReadyStatus,
		Input:  &input,
	}
	jobConfig.TriggerTime = time.Now()
	return &msg, true, nil
}

// triggerEvalTask triggers the eval task
func (lm *LifelongLearningJobManager) triggerEvalTask(job *LifelongLearningJob) (*UpstreamMessage, error) {
	jobConfig := job.JobConfig
	var err error

	var dataIndexURL string
	jobConfig.EvalDataURL, dataIndexURL, err = job.writeLLJSamples(jobConfig.DataSamples.EvalSamples, jobConfig.OutputConfig.SamplesOutput["eval"])
	if err != nil {
		klog.Errorf("job(name=%s) eval phase: write samples to the file(%s) is failed, error: %v",
			jobConfig.UniqueIdentifier, jobConfig.EvalDataURL, err)
		return nil, err
	}

	var models []ModelInfo
	models = append(models, ModelInfo{
		Format: jobConfig.TrainModel.Format,
		URL:    jobConfig.TrainModel.URL,
	})

	dataURL := jobConfig.EvalDataURL
	outputDir := strings.Join([]string{jobConfig.OutputConfig.EvalOutput, strconv.Itoa(jobConfig.Version)}, "/")
	if job.Storage.IsLocalStorage {
		dataURL = util.TrimPrefixPath(lm.VolumeMountPrefix, dataURL)
		dataIndexURL = util.TrimPrefixPath(lm.VolumeMountPrefix, dataIndexURL)
		outputDir = util.TrimPrefixPath(lm.VolumeMountPrefix, outputDir)
	}

	input := WorkerInput{
		Models:       models,
		DataURL:      dataURL,
		DataIndexURL: dataIndexURL,
		OutputDir:    outputDir,
	}
	msg := &UpstreamMessage{
		Phase:  EvalPhase,
		Status: WorkerReadyStatus,
		Input:  &input,
	}

	return msg, nil
}

// deployModel deploys model
func (lm *LifelongLearningJobManager) deployModel(job *LifelongLearningJob) (*ModelInfo, error) {
	jobConfig := job.JobConfig

	model := &ModelInfo{}
	model = jobConfig.EvalResult

	if job.Storage.IsLocalStorage {
		model.URL = util.AddPrefixPath(lm.VolumeMountPrefix, model.URL)
	}

	deployModelURL := jobConfig.DeployModel.URL
	if err := job.Storage.CopyFile(model.URL, deployModelURL); err != nil {
		return nil, fmt.Errorf("copy model(url=%s) to the deploy model(url=%s) failed, error: %+v",
			model.URL, deployModelURL, err)
	}
	klog.V(4).Infof("copy model(url=%s) to the deploy model(url=%s) successfully", model.URL, deployModelURL)

	klog.Infof("job(name=%s) deploys model(url=%s) successfully", jobConfig.UniqueIdentifier, model.URL)

	return model, nil
}

// createOutputDir creates the job output dir
func (job *LifelongLearningJob) createOutputDir(jobConfig *LLJobConfig) error {
	outputDir := jobConfig.OutputDir

	dirNames := []string{"data/train", "data/eval", "train", "eval"}
	// lifelong_kb_index.pkl

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

	outputConfig := LLOutputConfig{
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

// createFile creates data file and data index file
func (job *LifelongLearningJob) createFile(dir string, format string, isLocalStorage bool) (string, string) {
	switch strings.ToLower(format) {
	case DatasetFormatTXT:
		if isLocalStorage {
			return path.Join(dir, "data.txt"), ""
		}
		return strings.Join([]string{dir, "data.txt"}, "/"), strings.Join([]string{dir, "dataIndex.txt"}, "/")
	case DatasetFormatCSV:
		return strings.Join([]string{dir, "data.csv"}, "/"), ""
	}

	return "", ""
}

// writeLLJSamples writes samples information to a file
func (job *LifelongLearningJob) writeLLJSamples(samples []string, dir string) (string, string, error) {
	version := job.JobConfig.Version
	format := job.Dataset.Spec.Format
	urlPrefix := job.Dataset.URLPrefix

	subDir := strings.Join([]string{dir, strconv.Itoa(version)}, "/")
	fileURL, absURLFile := job.createFile(subDir, format, job.Dataset.Storage.IsLocalStorage)

	if job.Storage.IsLocalStorage {
		if err := util.CreateFolder(subDir); err != nil {
			return "", "", err
		}
		if err := job.writeByLine(samples, fileURL, format); err != nil {
			return "", "", err
		}

		if !job.Dataset.Storage.IsLocalStorage && absURLFile != "" {
			tempSamples := util.ParsingDatasetIndex(samples, urlPrefix)
			if err := job.writeByLine(tempSamples, absURLFile, format); err != nil {
				return "", "", err
			}
		}

		return fileURL, absURLFile, nil
	}

	temporaryDir, err := util.CreateTemporaryDir()
	if err != nil {
		return "", "", err
	}

	localFileURL, localAbsURLFile := job.createFile(temporaryDir, format, job.Dataset.Storage.IsLocalStorage)

	if err := job.writeByLine(samples, localFileURL, format); err != nil {
		return "", "", err
	}

	if err := job.Storage.Upload(localFileURL, fileURL); err != nil {
		return "", "", err
	}

	if absURLFile != "" {
		tempSamples := util.ParsingDatasetIndex(samples, urlPrefix)

		if err := job.writeByLine(tempSamples, localAbsURLFile, format); err != nil {
			return "", "", err
		}

		if err := job.Storage.Upload(localAbsURLFile, absURLFile); err != nil {
			return "", "", err
		}

		defer os.RemoveAll(localFileURL)
	}

	defer os.RemoveAll(localAbsURLFile)

	return fileURL, absURLFile, nil
}

// writeByLine writes file by line
func (job *LifelongLearningJob) writeByLine(samples []string, fileURL string, format string) error {
	file, err := os.Create(fileURL)
	if err != nil {
		klog.Errorf("create file(%s) failed", fileURL)
		return err
	}

	w := bufio.NewWriter(file)

	if format == "csv" {
		_, _ = fmt.Fprintln(w, job.Dataset.DataSource.Header)
	}

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

// handleData updates samples information
func (lm *LifelongLearningJobManager) handleData(job *LifelongLearningJob) {
	tick := time.NewTicker(LLHandlerDataIntervalSeconds * time.Second)

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
		err := lm.loadDataset(job)
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

func (lm *LifelongLearningJobManager) loadDataset(job *LifelongLearningJob) error {
	if job.Dataset != nil {
		// already loaded
		return nil
	}

	datasetName := util.GetUniqueIdentifier(job.Namespace, job.Spec.Dataset.Name, DatasetResourceKind)
	dataset, ok := lm.DatasetManager.GetDataset(datasetName)
	if !ok || dataset == nil {
		return fmt.Errorf("not exists dataset(name=%s)", datasetName)
	}

	jobConfig := job.JobConfig
	jobConfig.DataSamples = &LLDataSamples{
		Numbers:            0,
		TrainSamples:       make([]string, 0),
		EvalVersionSamples: make([][]string, 0),
		EvalSamples:        make([]string, 0),
	}

	job.Dataset = dataset
	return nil
}

// initJob inits the job object
func (lm *LifelongLearningJobManager) initJob(job *LifelongLearningJob) error {
	jobConfig := job.JobConfig
	jobConfig.TrainModel = new(ModelInfo)
	jobConfig.EvalResult = new(ModelInfo)
	jobConfig.Lock = sync.Mutex{}

	jobConfig.Version = 0
	jobConfig.Phase = TrainPhase
	jobConfig.WorkerStatus = WorkerReadyStatus
	jobConfig.TriggerStatus = TriggerReadyStatus
	trainTrigger, err := newLLTrigger(job.Spec.TrainSpec.Trigger)
	if err != nil {
		return fmt.Errorf("failed to init train trigger: %+w", err)
	}
	jobConfig.TrainTrigger = trainTrigger

	outputDir := job.Spec.OutputDir

	isLocalURL, err := job.Storage.IsLocalURL(outputDir)
	if err != nil {
		return fmt.Errorf("job(name=%s)'s output dir is invalid, error: %+v", job.Name, outputDir)
	}

	if isLocalURL {
		job.Storage.IsLocalStorage = true
		outputDir = util.AddPrefixPath(lm.VolumeMountPrefix, outputDir)
	}

	jobConfig.OutputDir = outputDir

	if err := job.createOutputDir(jobConfig); err != nil {
		return err
	}

	jobConfig.DeployModel = &ModelInfo{
		Format: "pkl",
		URL:    strings.Join([]string{strings.TrimRight(outputDir, "/"), "deploy/index.pkl"}, "/"),
	}

	return nil
}

func newLLTrigger(t sednav1.LLTrigger) (trigger.Base, error) {
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

// forwardSamplesLL deletes the samples information in the memory
func forwardSamplesLL(jobConfig *LLJobConfig) {
	switch jobConfig.Phase {
	case TrainPhase:
		{
			jobConfig.Lock.Lock()
			jobConfig.DataSamples.TrainSamples = jobConfig.DataSamples.TrainSamples[:0]
			jobConfig.Lock.Unlock()
		}
	case EvalPhase:
		{
			if len(jobConfig.DataSamples.EvalVersionSamples) > LLEvalSamplesCapacity {
				jobConfig.DataSamples.EvalVersionSamples = jobConfig.DataSamples.EvalVersionSamples[1:]
			}
		}
	}
}

// backLLTaskStatus backs train task status
func backLLTaskStatus(jobConfig *LLJobConfig) {
	jobConfig.Phase = TrainPhase
	initLLTaskStatus(jobConfig)
}

// initLLTaskStatus inits task status
func initLLTaskStatus(jobConfig *LLJobConfig) {
	jobConfig.WorkerStatus = WorkerReadyStatus
	jobConfig.TriggerStatus = TriggerReadyStatus
}

// nextLLTask converts next task status
func nextLLTask(jobConfig *LLJobConfig) {
	switch jobConfig.Phase {
	case TrainPhase:
		{
			forwardSamplesLL(jobConfig)
			initLLTaskStatus(jobConfig)
			jobConfig.Phase = EvalPhase
		}

	case EvalPhase:
		{
			forwardSamplesLL(jobConfig)
			initLLTaskStatus(jobConfig)
			jobConfig.Phase = DeployPhase
		}
	case DeployPhase:
		{
			backLLTaskStatus(jobConfig)
		}
	}
}

// Delete deletes lifelong-learning-job config in db
func (lm *LifelongLearningJobManager) Delete(message *gmclient.Message) error {
	name := util.GetUniqueIdentifier(message.Header.Namespace, message.Header.ResourceName, message.Header.ResourceKind)

	if job, ok := lm.LifelongLearningJobMap[name]; ok && job.Done != nil {
		close(job.Done)
	}

	delete(lm.LifelongLearningJobMap, name)

	if err := db.DeleteResource(name); err != nil {
		return err
	}

	return nil
}

// Start starts LifelongLearningJob manager
func (lm *LifelongLearningJobManager) Start() error {
	go lm.monitorWorker()

	return nil
}

// monitorWorker monitors message from worker
func (lm *LifelongLearningJobManager) monitorWorker() {
	for {
		workerMessageChannel := lm.WorkerMessageChannel
		workerMessage, ok := <-workerMessageChannel
		if !ok {
			break
		}
		klog.V(4).Infof("handling worker message %+v", workerMessage)

		name := util.GetUniqueIdentifier(workerMessage.Namespace, workerMessage.OwnerName, workerMessage.OwnerKind)

		job, ok := lm.LifelongLearningJobMap[name]
		if !ok {
			continue
		}

		// TODO: filter some worker messages out
		wo := WorkerOutput{}
		wo.Models = workerMessage.Results
		wo.OwnerInfo = workerMessage.OwnerInfo

		msg := &UpstreamMessage{
			Phase:  workerMessage.Kind,
			Status: workerMessage.Status,
			Output: &wo,
		}
		lm.Client.WriteMessage(msg, job.getHeader())

		lm.handleWorkerMessage(job, workerMessage)
	}
}

// handleWorkerMessage handles message from worker
func (lm *LifelongLearningJobManager) handleWorkerMessage(job *LifelongLearningJob, workerMessage WorkerMessage) {
	jobPhase := job.JobConfig.Phase
	workerKind := workerMessage.Kind
	if jobPhase != workerKind {
		klog.Warningf("job(name=%s) %s phase get worker(kind=%s)", job.JobConfig.UniqueIdentifier,
			jobPhase, workerKind)
		return
	}

	var models []*ModelInfo
	for _, result := range workerMessage.Results {
		model := ModelInfo{
			Format: result["format"].(string),
			URL:    result["url"].(string)}
		models = append(models, &model)
	}

	model := &ModelInfo{}
	if len(models) != 1 {
		return
	}
	model = models[0]

	job.JobConfig.WorkerStatus = workerMessage.Status

	if job.JobConfig.WorkerStatus == WorkerCompletedStatus {
		switch job.JobConfig.Phase {
		case TrainPhase:
			job.JobConfig.TrainModel = model
		case EvalPhase:
			job.JobConfig.EvalResult = model
		}
	}
}

// AddWorkerMessage adds worker messages
func (lm *LifelongLearningJobManager) AddWorkerMessage(message WorkerMessage) {
	lm.WorkerMessageChannel <- message
}

// GetName returns name of the manager
func (lm *LifelongLearningJobManager) GetName() string {
	return LifelongLearningJobKind
}

func (job *LifelongLearningJob) getHeader() gmclient.MessageHeader {
	return gmclient.MessageHeader{
		Namespace:    job.Namespace,
		ResourceKind: job.Kind,
		ResourceName: job.Name,
		Operation:    gmclient.StatusOperation,
	}
}
