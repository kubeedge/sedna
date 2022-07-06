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
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path"
	"strconv"
	"strings"
	"sync"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"

	"github.com/kubeedge/sedna/cmd/sedna-lc/app/options"
	sednav1 "github.com/kubeedge/sedna/pkg/apis/sedna/v1alpha1"
	gmtypes "github.com/kubeedge/sedna/pkg/globalmanager/controllers/lifelonglearning"
	"github.com/kubeedge/sedna/pkg/globalmanager/runtime"
	"github.com/kubeedge/sedna/pkg/localcontroller/db"
	clienttypes "github.com/kubeedge/sedna/pkg/localcontroller/gmclient"
	"github.com/kubeedge/sedna/pkg/localcontroller/managers/dataset"
	"github.com/kubeedge/sedna/pkg/localcontroller/storage"
	"github.com/kubeedge/sedna/pkg/localcontroller/trigger"
	"github.com/kubeedge/sedna/pkg/localcontroller/util"
	workertypes "github.com/kubeedge/sedna/pkg/localcontroller/worker"
	"github.com/microcosm-cc/bluemonday"
)

const (
	// JobIterationIntervalSeconds is interval time of each iteration of job
	JobIterationIntervalSeconds = 10
	// DatasetHandlerIntervalSeconds is interval time of handling dataset
	DatasetHandlerIntervalSeconds = 10
	// EvalSamplesCapacity is capacity of eval samples
	EvalSamplesCapacity = 5
	//KindName is kind of lifelong-learning-job resource
	KindName = "lifelonglearningjob"

	// TriggerReadyStatus is the ready status about trigger
	TriggerReadyStatus = "ready"
	// TriggerCompletedStatus is the completed status about trigger
	TriggerCompletedStatus = "completed"

	AnnotationsRoundsKey          = "sedna.io/rounds"
	AnnotationsNumberOfSamplesKey = "sedna.io/number-of-samples"
	AnnotationsDataFileOfEvalKey  = "sedna.io/data-file-of-eval"
)

// LifelongLearningJobManager defines lifelong-learning-job Manager
type Manager struct {
	Client                 clienttypes.ClientI
	WorkerMessageChannel   chan workertypes.MessageContent
	DatasetManager         *dataset.Manager
	LifelongLearningJobMap map[string]*Job
	VolumeMountPrefix      string
}

// LifelongLearningJob defines config for lifelong-learning-job
type Job struct {
	sednav1.LifelongLearningJob
	JobConfig *JobConfig
}

// JobConfig defines config for lifelong-learning-job
type JobConfig struct {
	UniqueIdentifier    string
	Rounds              int
	TrainTrigger        trigger.Base
	TriggerTime         time.Time
	TrainTriggerStatus  string
	EvalTriggerStatus   string
	DeployTriggerStatus string
	TrainDataURL        string
	EvalDataURL         string
	OutputDir           string
	OutputConfig        *OutputConfig
	DataSamples         *DataSamples
	DeployModel         *Model
	Lock                sync.Mutex
	Dataset             *dataset.Dataset
	Storage             storage.Storage
	Done                chan struct{}
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
	PreviousNumbers    int
	TrainSamples       []string
	EvalVersionSamples [][]string
	EvalSamples        []string
}

// New creates a lifelong-learning-job manager
func New(client clienttypes.ClientI, datasetManager *dataset.Manager, options *options.LocalControllerOptions) *Manager {
	lm := Manager{
		Client:                 client,
		WorkerMessageChannel:   make(chan workertypes.MessageContent, workertypes.MessageChannelCacheSize),
		DatasetManager:         datasetManager,
		LifelongLearningJobMap: make(map[string]*Job),
		VolumeMountPrefix:      options.VolumeMountPrefix,
	}

	return &lm
}

// Insert inserts lifelong-learning-job config to db
func (lm *Manager) Insert(message *clienttypes.Message) error {
	p := bluemonday.NewPolicy()
	name := p.Sanitize(util.GetUniqueIdentifier(message.Header.Namespace,
		message.Header.ResourceName, message.Header.ResourceKind))

	first := false
	job, ok := lm.LifelongLearningJobMap[name]
	if !ok {
		job = &Job{}
		lm.LifelongLearningJobMap[name] = job
		first = true
	}

	if err := json.Unmarshal(message.Content, &job); err != nil {
		return err
	}

	if err := db.SaveResource(name, job.TypeMeta, job.ObjectMeta, job.Spec); err != nil {
		return err
	}

	if first {
		go lm.startJob(name)
	}

	return nil
}

// startJob starts a job
func (lm *Manager) startJob(name string) {
	var err error
	job, ok := lm.LifelongLearningJobMap[name]
	if !ok {
		return
	}

	if err = lm.initJob(job, name); err != nil {
		klog.Errorf("failed to init job(%s): %+v", name)
		return
	}

	klog.Infof("lifelong learning job(%s) is started", name)
	defer klog.Infof("lifelong learning job(%s) is stopped", name)

	// handle data from dataset
	go lm.handleData(job)

	tick := time.NewTicker(JobIterationIntervalSeconds * time.Second)
	for {
		select {
		case <-job.JobConfig.Done:
			return
		case <-tick.C:
			cond := lm.getLatestCondition(job)
			jobStage := cond.Stage

			switch jobStage {
			case sednav1.LLJobTrain:
				err = lm.trainTask(job)
			case sednav1.LLJobEval:
				err = lm.evalTask(job)

			case sednav1.LLJobDeploy:
				err = lm.deployTask(job)
			default:
				klog.Errorf("invalid phase: %s", jobStage)
				continue
			}

			if err != nil {
				klog.Errorf("job(%s) failed to complete the %s task: %v", name, jobStage, err)
			}
		}
	}
}

// trainTask starts training task
func (lm *Manager) trainTask(job *Job) error {
	jobConfig := job.JobConfig

	latestCond := lm.getLatestCondition(job)
	jobStage := latestCond.Stage
	currentType := latestCond.Type

	if currentType == sednav1.LLJobStageCondWaiting {
		err := lm.loadDataset(job)
		if err != nil || jobConfig.Dataset == nil || jobConfig.Dataset.DataSource == nil {
			return fmt.Errorf("job(%s) failed to load dataset, and waiting it: %w",
				jobConfig.UniqueIdentifier, err)
		}

		if jobConfig.Dataset == nil || jobConfig.Dataset.DataSource == nil {
			return fmt.Errorf("job(%s)'s dataset not ready", jobConfig.UniqueIdentifier)
		}

		initTriggerStatus(jobConfig)

		if jobConfig.TrainTriggerStatus == TriggerReadyStatus {
			payload, ok, err := lm.triggerTrainTask(job)
			if !ok {
				return nil
			}

			if err != nil {
				klog.Errorf("job(%s) failed to complete the %sing phase triggering task: %v",
					jobConfig.UniqueIdentifier, jobStage, err)
				job.JobConfig.Rounds--
				return err
			}

			err = lm.Client.WriteMessage(payload, job.getHeader())
			if err != nil {
				klog.Errorf("job(%s) failed to write message: %v", jobConfig.UniqueIdentifier, err)
				job.JobConfig.Rounds--
				return err
			}

			forwardSamples(jobConfig, jobStage)

			err = lm.saveJobToDB(job)
			if err != nil {
				klog.Errorf("job(%s) failed to save job to db: %v",
					jobConfig.UniqueIdentifier, err)
				// continue anyway
			}

			jobConfig.TrainTriggerStatus = TriggerCompletedStatus
			klog.Infof("job(name=%s) complete the %sing phase triggering task successfully",
				jobConfig.UniqueIdentifier, jobStage)
		}
	}

	return nil
}

// evalTask starts eval task
func (lm *Manager) evalTask(job *Job) error {
	jobConfig := job.JobConfig

	latestCond := lm.getLatestCondition(job)
	jobStage := latestCond.Stage
	currentType := latestCond.Type

	if currentType == sednav1.LLJobStageCondWaiting {
		err := lm.loadDataset(job)
		if err != nil || jobConfig.Dataset == nil || jobConfig.Dataset.DataSource == nil {
			return fmt.Errorf("job(%s) failed to load dataset, and waiting it: %w",
				jobConfig.UniqueIdentifier, err)
		}

		if jobConfig.EvalTriggerStatus == TriggerReadyStatus {
			payload, err := lm.triggerEvalTask(job)
			if err != nil {
				klog.Errorf("job(%s) completed the %sing phase triggering task failed: %v",
					jobConfig.UniqueIdentifier, jobStage, err)
				return err
			}

			err = lm.Client.WriteMessage(payload, job.getHeader())
			if err != nil {
				klog.Errorf("job(%s) failed to write message: %v", jobConfig.UniqueIdentifier, err)
				return err
			}

			forwardSamples(jobConfig, jobStage)

			jobConfig.EvalTriggerStatus = TriggerCompletedStatus
			klog.Infof("job(%s) completed the %sing phase triggering task successfully",
				jobConfig.UniqueIdentifier, jobStage)
		}
	}

	return nil
}

// deployTask starts deploy task
func (lm *Manager) deployTask(job *Job) error {
	if job.JobConfig.DeployTriggerStatus == TriggerReadyStatus {
		jobConfig := job.JobConfig
		var err error

		status := clienttypes.UpstreamMessage{Phase: string(sednav1.LLJobDeploy)}
		models := lm.getJobStageModel(job, sednav1.LLJobDeploy)
		if models != nil {
			err = lm.updateDeployModelFile(job, models[0].URL, jobConfig.DeployModel.URL)
			if err != nil {
				status.Status = string(sednav1.LLJobStageCondFailed)
				klog.Errorf("failed to update model for job(%s): %v", jobConfig.UniqueIdentifier, err)
				return err
			}

			status.Status = string(sednav1.LLJobStageCondReady)
			status.Input = &clienttypes.Input{Models: []Model{{Format: models[0].Format, URL: models[0].URL}}}
		} else {
			klog.Infof("job(%s) isn't need to deploy model", jobConfig.UniqueIdentifier)
			status.Status = string(sednav1.LLJobStageCondCompleted)
		}

		err = lm.Client.WriteMessage(status, job.getHeader())
		if err != nil {
			klog.Errorf("job(%s) completed the %s task failed: %v",
				jobConfig.UniqueIdentifier, sednav1.LLJobDeploy, err)
			return err
		}

		job.JobConfig.DeployTriggerStatus = TriggerCompletedStatus
		klog.Infof("job(%s) completed the %s task successfully", jobConfig.UniqueIdentifier, sednav1.LLJobDeploy)
	}
	return nil
}

// triggerTrainTask triggers the train task
func (lm *Manager) triggerTrainTask(job *Job) (interface{}, bool, error) {
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

	job.JobConfig.Rounds++
	rounds := jobConfig.Rounds

	var dataIndexURL string
	jobConfig.TrainDataURL, dataIndexURL, err = lm.writeSamples(job, jobConfig.DataSamples.TrainSamples,
		jobConfig.OutputConfig.SamplesOutput["train"], rounds, jobConfig.Dataset.Spec.Format, jobConfig.Dataset.URLPrefix)
	if err != nil {
		job.JobConfig.Rounds--
		klog.Errorf("job(%s) train phase: write samples to the file(%s) is failed: %v",
			jobConfig.UniqueIdentifier, jobConfig.TrainDataURL, err)
		return nil, false, err
	}

	dataURL := jobConfig.TrainDataURL
	outputDir := strings.Join([]string{jobConfig.OutputConfig.TrainOutput, strconv.Itoa(rounds)}, "/")
	if jobConfig.Storage.IsLocalStorage {
		dataURL = util.TrimPrefixPath(lm.VolumeMountPrefix, dataURL)
		dataIndexURL = util.TrimPrefixPath(lm.VolumeMountPrefix, dataIndexURL)
		outputDir = util.TrimPrefixPath(lm.VolumeMountPrefix, outputDir)
	}

	input := clienttypes.Input{
		DataURL:      dataURL,
		DataIndexURL: dataIndexURL,
		OutputDir:    outputDir,
	}
	msg := clienttypes.UpstreamMessage{
		Phase:  string(sednav1.LLJobTrain),
		Status: string(sednav1.LLJobStageCondReady),
		Input:  &input,
	}

	jobConfig.TriggerTime = time.Now()
	return &msg, true, nil
}

// triggerEvalTask triggers the eval task
func (lm *Manager) triggerEvalTask(job *Job) (*clienttypes.UpstreamMessage, error) {
	jobConfig := job.JobConfig
	var err error

	latestCondition := lm.getLatestCondition(job)

	ms := lm.getJobStageModel(job, latestCondition.Stage)
	if ms == nil {
		return nil, err
	}

	var dataIndexURL string
	jobConfig.EvalDataURL, dataIndexURL, err = lm.writeSamples(job, jobConfig.DataSamples.EvalSamples, jobConfig.OutputConfig.SamplesOutput["eval"],
		job.JobConfig.Rounds, jobConfig.Dataset.Spec.Format, jobConfig.Dataset.URLPrefix)
	if err != nil {
		klog.Errorf("job(%s) eval phase: write samples to the file(%s) is failed: %v",
			jobConfig.UniqueIdentifier, jobConfig.EvalDataURL, err)
		return nil, err
	}

	dataURL := jobConfig.EvalDataURL
	outputDir := strings.Join([]string{jobConfig.OutputConfig.EvalOutput, strconv.Itoa(jobConfig.Rounds)}, "/")
	if jobConfig.Storage.IsLocalStorage {
		dataURL = util.TrimPrefixPath(lm.VolumeMountPrefix, dataURL)
		dataIndexURL = util.TrimPrefixPath(lm.VolumeMountPrefix, dataIndexURL)
		outputDir = util.TrimPrefixPath(lm.VolumeMountPrefix, outputDir)
	}

	input := clienttypes.Input{
		Models:       ms,
		DataURL:      dataURL,
		DataIndexURL: dataIndexURL,
		OutputDir:    outputDir,
	}
	msg := &clienttypes.UpstreamMessage{
		Phase:  string(sednav1.LLJobEval),
		Status: string(sednav1.LLJobStageCondReady),
		Input:  &input,
	}

	return msg, nil
}

// updateDeployModelFile updates deploy model file
func (lm *Manager) updateDeployModelFile(job *Job, trainedModel string, deployModel string) error {
	if job.JobConfig.Storage.IsLocalStorage {
		trainedModel = util.AddPrefixPath(lm.VolumeMountPrefix, trainedModel)
	}

	if err := job.JobConfig.Storage.CopyFile(trainedModel, deployModel); err != nil {
		return fmt.Errorf("failed to copy trained model(url=%s) to the deploy model(url=%s): %w",
			trainedModel, deployModel, err)
	}

	klog.V(4).Infof("copy trained model(url=%s) to the deploy model(url=%s) successfully", trainedModel, deployModel)

	return nil
}

// createOutputDir creates the job output dir
func (lm *Manager) createOutputDir(jobConfig *JobConfig) error {
	outputDir := jobConfig.OutputDir

	dirNames := []string{"data/train", "data/eval", "train", "eval"}

	if jobConfig.Storage.IsLocalStorage {
		if err := util.CreateFolder(outputDir); err != nil {
			return fmt.Errorf("failed to create folder %s: %v", outputDir, err)
		}

		for _, v := range dirNames {
			dir := path.Join(outputDir, v)
			if err := util.CreateFolder(dir); err != nil {
				return fmt.Errorf("failed to create folder %s: %v", dir, err)
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

func (lm *Manager) getLatestCondition(job *Job) sednav1.LLJobCondition {
	jobConditions := job.Status.Conditions
	var latestCondition sednav1.LLJobCondition = sednav1.LLJobCondition{}
	if len(jobConditions) > 0 {
		// get latest pod and pod status
		latestCondition = jobConditions[len(jobConditions)-1]
	}
	return latestCondition
}

// createFile creates data file and data index file
func createFile(dir string, format string, isLocalStorage bool) (string, string) {
	switch strings.ToLower(format) {
	case dataset.TXTFormat:
		if isLocalStorage {
			return path.Join(dir, "data.txt"), ""
		}
		return strings.Join([]string{dir, "data.txt"}, "/"), strings.Join([]string{dir, "dataIndex.txt"}, "/")
	case dataset.CSVFormat:
		return strings.Join([]string{dir, "data.csv"}, "/"), ""
	}

	return "", ""
}

// writeSamples writes samples information to a file
func (lm *Manager) writeSamples(job *Job, samples []string, dir string, rounds int, format string, urlPrefix string) (string, string, error) {
	if samples == nil {
		return "", "", fmt.Errorf("not samples")
	}

	jobConfig := job.JobConfig
	subDir := strings.Join([]string{dir, strconv.Itoa(rounds)}, "/")
	fileURL, absURLFile := createFile(subDir, format, jobConfig.Dataset.Storage.IsLocalStorage)

	if jobConfig.Storage.IsLocalStorage {
		if err := util.CreateFolder(subDir); err != nil {
			return "", "", err
		}
		if err := job.writeByLine(samples, fileURL, format); err != nil {
			return "", "", err
		}

		if !jobConfig.Dataset.Storage.IsLocalStorage && absURLFile != "" {
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

	localFileURL, localAbsURLFile := createFile(temporaryDir, format, jobConfig.Dataset.Storage.IsLocalStorage)

	if err := job.writeByLine(samples, localFileURL, format); err != nil {
		return "", "", err
	}

	if err := jobConfig.Storage.Upload(localFileURL, fileURL); err != nil {
		return "", "", err
	}

	if absURLFile != "" {
		tempSamples := util.ParsingDatasetIndex(samples, urlPrefix)

		if err := job.writeByLine(tempSamples, localAbsURLFile, format); err != nil {
			return "", "", err
		}

		if err := jobConfig.Storage.Upload(localAbsURLFile, absURLFile); err != nil {
			return "", "", err
		}

		defer os.RemoveAll(localFileURL)
	}

	defer os.RemoveAll(localAbsURLFile)

	return fileURL, absURLFile, nil
}

// writeByLine writes file by line
func (job *Job) writeByLine(samples []string, fileURL string, format string) error {
	file, err := os.Create(fileURL)
	if err != nil {
		klog.Errorf("create file(%s) failed", fileURL)
		return err
	}

	w := bufio.NewWriter(file)

	if format == "csv" {
		_, _ = fmt.Fprintln(w, job.JobConfig.Dataset.DataSource.Header)
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
func (lm *Manager) handleData(job *Job) {
	tick := time.NewTicker(DatasetHandlerIntervalSeconds * time.Second)

	jobConfig := job.JobConfig
	iterCount := 0
	for {
		select {
		case <-jobConfig.Done:
			return
		default:
		}

		if iterCount%100 == 0 {
			klog.V(4).Infof("job(%s) is handling dataset", jobConfig.UniqueIdentifier)
		}
		iterCount++

		if jobConfig.Dataset == nil || jobConfig.Dataset.DataSource == nil {
			// already loaded dataset
			<-tick.C
			continue
		}

		dataset := jobConfig.Dataset
		currentNumberOfSamples := dataset.DataSource.NumberOfSamples
		previousNumberOfSamples := jobConfig.DataSamples.PreviousNumbers

		if dataset.DataSource != nil && currentNumberOfSamples > previousNumberOfSamples {
			samples := dataset.DataSource.TrainSamples
			newNumberOfSamples := currentNumberOfSamples - previousNumberOfSamples
			trainNum := int(job.Spec.Dataset.TrainProb * float64(newNumberOfSamples))

			jobConfig.Lock.Lock()
			jobConfig.DataSamples.TrainSamples = append(jobConfig.DataSamples.TrainSamples,
				samples[previousNumberOfSamples:previousNumberOfSamples+trainNum]...)
			klog.Infof("job(%s)'s current train samples nums is %d", jobConfig.UniqueIdentifier, trainNum)

			jobConfig.DataSamples.EvalVersionSamples = append(jobConfig.DataSamples.EvalVersionSamples,
				samples[previousNumberOfSamples+trainNum:])
			jobConfig.Lock.Unlock()

			for _, v := range jobConfig.DataSamples.EvalVersionSamples {
				jobConfig.DataSamples.EvalSamples = append(jobConfig.DataSamples.EvalSamples, v...)
			}
			klog.Infof("job(%s)'s current eval samples nums is %d", jobConfig.UniqueIdentifier, len(jobConfig.DataSamples.EvalSamples))

			jobConfig.DataSamples.PreviousNumbers = currentNumberOfSamples
		}

		<-tick.C
	}
}

// loadDataset loads dataset information
func (lm *Manager) loadDataset(job *Job) error {
	if job.JobConfig.Dataset != nil {
		// already loaded
		return nil
	}

	datasetName := util.GetUniqueIdentifier(job.Namespace, job.Spec.Dataset.Name, dataset.KindName)
	dataset, ok := lm.DatasetManager.GetDataset(datasetName)
	if !ok || dataset == nil {
		return fmt.Errorf("not exists dataset(name=%s)", datasetName)
	}

	job.JobConfig.Dataset = dataset
	return nil
}

// initJob inits the job object
func (lm *Manager) initJob(job *Job, name string) error {
	var err error
	jobConfig := new(JobConfig)

	jobConfig.UniqueIdentifier = name

	jobConfig.Storage = storage.Storage{IsLocalStorage: false}
	credential := job.ObjectMeta.Annotations[runtime.SecretAnnotationKey]
	if credential != "" {
		if err := jobConfig.Storage.SetCredential(credential); err != nil {
			return fmt.Errorf("failed to set storage credential: %w", err)
		}
	}

	jobConfig.Done = make(chan struct{})
	jobConfig.Lock = sync.Mutex{}
	jobConfig.Rounds = 0

	jobConfig.DataSamples = &DataSamples{
		PreviousNumbers:    0,
		TrainSamples:       make([]string, 0),
		EvalVersionSamples: make([][]string, 0),
		EvalSamples:        make([]string, 0),
	}

	trainTrigger, err := newTrigger(job.Spec.TrainSpec.Trigger)
	if err != nil {
		return fmt.Errorf("failed to init train trigger: %+w", err)
	}
	jobConfig.TrainTrigger = trainTrigger

	outputDir := job.Spec.OutputDir

	isLocalURL, err := jobConfig.Storage.IsLocalURL(outputDir)
	if err != nil {
		return fmt.Errorf("job(%s)'s output dir(%s) is invalid: %+w", name, outputDir, err)
	}

	if isLocalURL {
		jobConfig.Storage.IsLocalStorage = true
		outputDir = util.AddPrefixPath(lm.VolumeMountPrefix, outputDir)
	}

	jobConfig.OutputDir = outputDir

	if err = lm.createOutputDir(jobConfig); err != nil {
		return err
	}

	jobConfig.DeployModel = &Model{
		Format: "pkl",
		URL:    strings.Join([]string{strings.TrimRight(outputDir, "/"), "deploy/index.pkl"}, "/"),
	}

	initTriggerStatus(jobConfig)

	job.JobConfig = jobConfig

	if err = lm.updateJobFromDB(job); err != nil {
		return fmt.Errorf("failed to update job from db, error: %v", err)
	}

	return nil
}

func initTriggerStatus(jobConfig *JobConfig) {
	jobConfig.TrainTriggerStatus = TriggerReadyStatus
	jobConfig.EvalTriggerStatus = TriggerReadyStatus
	jobConfig.DeployTriggerStatus = TriggerReadyStatus
}

func newTrigger(t sednav1.LLTrigger) (trigger.Base, error) {
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

// getModelsFromJobConditions gets models from job condition
func (lm *Manager) getModelsFromJobConditions(jobConditions []sednav1.LLJobCondition, stage sednav1.LLJobStage, currentType sednav1.LLJobStageConditionType, dataType string) []Model {
	// TODO: runtime.type changes to common.type for gm and lc
	for i := len(jobConditions) - 1; i >= 0; i-- {
		var cond gmtypes.ConditionData
		jobCond := jobConditions[i]
		if jobCond.Stage == stage && jobCond.Type == currentType {
			if err := (&cond).Unmarshal([]byte(jobCond.Data)); err != nil {
				continue
			}

			if dataType == "input" {
				if cond.Input == nil {
					continue
				}

				return cond.Input.Models
			} else if dataType == "output" {
				if cond.Output == nil {
					continue
				}

				return cond.Output.Models
			}
		}
	}

	return nil
}

// getEvalResult gets eval result from job conditions
func (lm *Manager) getEvalResult(job *Job) ([]map[string][]float64, error) {
	jobConditions := job.Status.Conditions
	models := lm.getModelsFromJobConditions(jobConditions, sednav1.LLJobEval, sednav1.LLJobStageCondCompleted, "output")

	var result []map[string][]float64
	var err error
	for _, m := range models {
		bytes, err := json.Marshal(m.Metrics)
		if err != nil {
			return nil, err
		}

		data := make(map[string][]float64)
		if err = json.Unmarshal(bytes, &data); err != nil {
			return nil, err
		}

		result = append(result, data)
	}
	return result, err
}

// getJobStageModel gets model from job conditions for eval/deploy
func (lm *Manager) getJobStageModel(job *Job, jobStage sednav1.LLJobStage) (models []Model) {
	jobConditions := job.Status.Conditions

	switch jobStage {
	case sednav1.LLJobEval:
		models = lm.getModelsFromJobConditions(jobConditions, sednav1.LLJobTrain, sednav1.LLJobStageCondCompleted, "output")
	case sednav1.LLJobDeploy:
		models = lm.getModelsFromJobConditions(jobConditions, sednav1.LLJobEval, sednav1.LLJobStageCondCompleted, "output")
	}

	return models
}

// forwardSamples deletes the samples information in the memory
func forwardSamples(jobConfig *JobConfig, jobStage sednav1.LLJobStage) {
	switch jobStage {
	case sednav1.LLJobTrain:
		jobConfig.Lock.Lock()
		jobConfig.DataSamples.TrainSamples = jobConfig.DataSamples.TrainSamples[:0]
		jobConfig.Lock.Unlock()
	case sednav1.LLJobEval:
		if len(jobConfig.DataSamples.EvalVersionSamples) > EvalSamplesCapacity {
			jobConfig.DataSamples.EvalVersionSamples = jobConfig.DataSamples.EvalVersionSamples[1:]
		}
	}
}

// Delete deletes lifelong-learning-job config in db
func (lm *Manager) Delete(message *clienttypes.Message) error {
	name := util.GetUniqueIdentifier(message.Header.Namespace, message.Header.ResourceName, message.Header.ResourceKind)

	if job, ok := lm.LifelongLearningJobMap[name]; ok && job.JobConfig.Done != nil {
		close(job.JobConfig.Done)
	}

	delete(lm.LifelongLearningJobMap, name)

	if err := db.DeleteResource(name); err != nil {
		return err
	}

	return nil
}

// updateJobFromDB updates job from db
func (lm *Manager) updateJobFromDB(job *Job) error {
	var err error

	previousJob, err := db.GetResource(job.JobConfig.UniqueIdentifier)
	if err != nil {
		return err
	}

	m := metav1.ObjectMeta{}
	if err != json.Unmarshal([]byte(previousJob.ObjectMeta), &m) {
		return err
	}

	rounds, ok := m.Annotations[AnnotationsRoundsKey]
	if !ok {
		return nil
	}

	if job.JobConfig.Rounds, err = strconv.Atoi(rounds); err != nil {
		return err
	}

	numberOfSamples, ok := m.Annotations[AnnotationsNumberOfSamplesKey]
	if !ok {
		return nil
	}

	if job.JobConfig.DataSamples.PreviousNumbers, err = strconv.Atoi(numberOfSamples); err != nil {
		return err
	}

	dataFileOfEval, ok := m.Annotations[AnnotationsDataFileOfEvalKey]
	if !ok {
		return nil
	}

	localURL, err := job.JobConfig.Storage.Download(dataFileOfEval, "")

	if !job.JobConfig.Storage.IsLocalStorage {
		defer os.RemoveAll(localURL)
	}

	if err != nil {
		return err
	}

	samples, err := dataset.GetSamples(dataFileOfEval)
	if err != nil {
		klog.Errorf("read file %s failed: %v", dataFileOfEval, err)
		return err
	}

	job.JobConfig.DataSamples.EvalVersionSamples = append(job.JobConfig.DataSamples.EvalVersionSamples, samples)

	return nil
}

// saveJobToDB saves job info to db
func (lm *Manager) saveJobToDB(job *Job) error {
	ann := job.ObjectMeta.Annotations
	if ann == nil {
		ann = make(map[string]string)
	}

	ann[AnnotationsRoundsKey] = strconv.Itoa(job.JobConfig.Rounds)
	ann[AnnotationsNumberOfSamplesKey] = strconv.Itoa(job.JobConfig.DataSamples.PreviousNumbers)
	ann[AnnotationsDataFileOfEvalKey] = job.JobConfig.EvalDataURL

	return db.SaveResource(job.JobConfig.UniqueIdentifier, job.TypeMeta, job.ObjectMeta, job.Spec)
}

// Start starts lifelong-learning-job manager
func (lm *Manager) Start() error {
	go lm.monitorWorker()

	return nil
}

// monitorWorker monitors message from worker
func (lm *Manager) monitorWorker() {
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
		wo := clienttypes.Output{}
		wo.Models = workerMessage.Results
		wo.OwnerInfo = workerMessage.OwnerInfo

		msg := &clienttypes.UpstreamMessage{
			Phase:  workerMessage.Kind,
			Status: workerMessage.Status,
			Output: &wo,
		}
		if err := lm.Client.WriteMessage(msg, job.getHeader()); err != nil {
			klog.Errorf("job(%s) failed to write message: %v", name, err)
			continue
		}
	}
}

// AddWorkerMessage adds worker messages
func (lm *Manager) AddWorkerMessage(message workertypes.MessageContent) {
	lm.WorkerMessageChannel <- message
}

// GetName returns name of the manager
func (lm *Manager) GetName() string {
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
