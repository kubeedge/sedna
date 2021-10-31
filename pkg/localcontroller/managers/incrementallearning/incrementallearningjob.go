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
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// IncrementalLearningJob defines config for incremental-learning-job
type Job struct {
	sednav1.IncrementalLearningJob
	JobConfig *JobConfig
}

// JobConfig defines config for incremental-learning-job
type JobConfig struct {
	UniqueIdentifier           string
	Rounds                     int
	TrainTrigger               trigger.Base
	DeployTrigger              trigger.Base
	TriggerTime                time.Time
	TrainTriggerStatus         string
	EvalTriggerStatus          string
	DeployTriggerStatus        string
	HotModelUpdateDeployStatus string
	TrainDataURL               string
	EvalDataURL                string
	OutputDir                  string
	OutputConfig               *OutputConfig
	DataSamples                *DataSamples
	TrainModel                 *Model
	DeployModel                *Model
	EvalModel                  *Model
	EvalResult                 []Model
	Lock                       sync.Mutex
	Dataset                    *dataset.Dataset
	Storage                    storage.Storage
	Done                       chan struct{}
}

type Model = runtime.Model

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

	AnnotationsRoundsKey          = "sedna.io/rounds"
	AnnotationsNumberOfSamplesKey = "sedna.io/number-of-samples"
	AnnotationsDataFileOfEvalKey  = "sedna.io/data-file-of-eval"
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
func (im *Manager) trainTask(job *Job) error {
	jobConfig := job.JobConfig

	latestCond := im.getLatestCondition(job)
	jobStage := latestCond.Stage
	currentType := latestCond.Type

	if currentType == sednav1.ILJobStageCondWaiting {
		var err error

		err = im.loadDataset(job)
		if err != nil || jobConfig.Dataset == nil || jobConfig.Dataset.DataSource == nil {
			return fmt.Errorf("job(%s) failed to load dataset, and waiting it: %w",
				jobConfig.UniqueIdentifier, err)
		}

		if jobConfig.Dataset == nil || jobConfig.Dataset.DataSource == nil {
			return fmt.Errorf("job(%s)'s dataset not ready", jobConfig.UniqueIdentifier)
		}

		err = im.loadTrainModel(job)
		if err != nil {
			return fmt.Errorf("failed to sync train model, and waiting it: %w", err)
		}

		initTriggerStatus(jobConfig)

		if jobConfig.TrainTriggerStatus == TriggerReadyStatus {
			payload, ok, err := im.triggerTrainTask(job)
			if !ok {
				return nil
			}

			if err != nil {
				klog.Errorf("job(%s) failed to complete the %sing phase triggering task: %v",
					jobConfig.UniqueIdentifier, jobStage, err)
				job.JobConfig.Rounds--
				return err
			}

			err = im.Client.WriteMessage(payload, job.getHeader())
			if err != nil {
				klog.Errorf("job(%s) failed to write message: %v", jobConfig.UniqueIdentifier, err)
				job.JobConfig.Rounds--
				return err
			}

			forwardSamples(jobConfig, jobStage)

			err = im.saveJobToDB(job)
			if err != nil {
				klog.Errorf("job(%s) failed to save job to db: %v",
					jobConfig.UniqueIdentifier, err)
				// continue anyway
			}

			jobConfig.TrainTriggerStatus = TriggerCompletedStatus
			klog.Infof("job(%s) completed the %sing phase triggering task successfully",
				jobConfig.UniqueIdentifier, jobStage)
		}
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
		var err error

		err = im.loadDataset(job)
		if err != nil || jobConfig.Dataset == nil || jobConfig.Dataset.DataSource == nil {
			return fmt.Errorf("job(%s) failed to load dataset, and waiting it: %w",
				jobConfig.UniqueIdentifier, err)
		}

		err = im.loadDeployModel(job)
		if err != nil {
			return fmt.Errorf("failed to sync deploy model, and waiting it: %w", err)
		}

		if job.Spec.EvalSpec.InitialModel != nil {
			err = im.loadEvalModel(job)
			if err != nil {
				return fmt.Errorf("failed to sync initial eval model, and waiting it: %w", err)
			}
		} else {
			if jobConfig.EvalModel == nil {
				jobConfig.EvalModel = jobConfig.DeployModel
			}
		}

		if jobConfig.EvalTriggerStatus == TriggerReadyStatus {
			payload, err := im.triggerEvalTask(job)
			if err != nil {
				klog.Errorf("job(%s) completed the %sing phase triggering task failed: %v",
					jobConfig.UniqueIdentifier, jobStage, err)
				return err
			}

			err = im.Client.WriteMessage(payload, job.getHeader())
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

// hotModelUpdateDeployTask starts deploy task when job supports hot model update
func (im *Manager) hotModelUpdateDeployTask(job *Job) error {
	var localModelConfigFile string
	if v, ok := job.ObjectMeta.Annotations[runtime.ModelHotUpdateAnnotationsKey]; ok {
		localModelConfigFile = v
	} else {
		return nil
	}

	if job.JobConfig.HotModelUpdateDeployStatus == TriggerReadyStatus {
		var err error
		err = im.loadDeployModel(job)
		if err != nil {
			return fmt.Errorf("failed to sync deploy model, and waiting it: %w", err)
		}

		if job.Spec.EvalSpec.InitialModel != nil {
			err = im.loadEvalModel(job)
			if err != nil {
				return fmt.Errorf("failed to sync initial eval model, and waiting it: %w", err)
			}
		}

		trainedModel := im.getModelFromJobConditions(job, sednav1.ILJobDeploy)
		deployModel := job.JobConfig.DeployModel

		trainedModelURL := trainedModel.URL
		modelName := filepath.Base(trainedModelURL)
		localHostDir := filepath.Dir(localModelConfigFile)
		localHostModelFile := filepath.Join(localHostDir, modelName)

		modelFile := util.AddPrefixPath(im.VolumeMountPrefix, localHostModelFile)
		if err := im.updateDeployModelFile(job, trainedModelURL, modelFile); err != nil {
			return err
		}

		deployModelURL := deployModel.URL
		if err := im.updateDeployModelFile(job, trainedModelURL, deployModelURL); err != nil {
			return err
		}

		evalModel := job.JobConfig.EvalModel
		if evalModel != nil {
			newEvalModel := im.getModelFromJobConditions(job, sednav1.ILJobEval)
			if err := im.updateDeployModelFile(job, newEvalModel.URL, evalModel.URL); err != nil {
				return err
			}
		}

		config := map[string]map[string]string{
			"model_config": {
				"model_path": strings.Replace(localHostModelFile, localHostDir,
					runtime.ModelHotUpdateContainerPrefix, 1),
				"model_update_time": time.Now().String(),
			},
		}

		jsonConfig, err := json.MarshalIndent(config, "", " ")
		if err != nil {
			return err
		}

		modelConfigFile := util.AddPrefixPath(im.VolumeMountPrefix, localModelConfigFile)
		// overwrite file
		err = ioutil.WriteFile(modelConfigFile, jsonConfig, 0644)
		if err != nil {
			klog.Errorf("job(%s) write model config file(url=%s) failed in deploy phase: %v",
				job.JobConfig.UniqueIdentifier, modelConfigFile, err)
			return err
		}

		job.JobConfig.HotModelUpdateDeployStatus = TriggerCompletedStatus
		klog.V(4).Infof("job(%s) write model config file(url=%s) successfully in deploy phase",
			job.JobConfig.UniqueIdentifier, modelConfigFile)
	}

	return nil
}

// deployTask starts deploy task
func (im *Manager) deployTask(job *Job) error {
	if job.JobConfig.DeployTriggerStatus == TriggerReadyStatus {
		if err := im.loadDeployModel(job); err != nil {
			return fmt.Errorf("failed to sync deploy model, and waiting it: %w", err)
		}

		if !job.Spec.DeploySpec.Model.HotUpdateEnabled && job.Spec.EvalSpec.InitialModel != nil {
			err := im.loadEvalModel(job)
			if err != nil {
				return fmt.Errorf("failed to sync initial eval model, and waiting it: %w", err)
			}
		}

		jobConfig := job.JobConfig
		var err error
		var neededDeploy bool

		neededDeploy, err = im.triggerDeployTask(job)
		status := clienttypes.UpstreamMessage{Phase: string(sednav1.ILJobDeploy)}

		if err == nil && neededDeploy {
			var models []Model
			trainedModel := im.getModelFromJobConditions(job, sednav1.ILJobDeploy)
			deployModel := jobConfig.DeployModel
			models = append(models, *trainedModel, *deployModel)

			if !job.Spec.DeploySpec.Model.HotUpdateEnabled {
				err = im.updateDeployModelFile(job, trainedModel.URL, deployModel.URL)
				if err != nil {
					status.Status = string(sednav1.ILJobStageCondFailed)
					klog.Errorf("failed to update model for job(%s): %v", jobConfig.UniqueIdentifier, err)
				} else {
					status.Status = string(sednav1.ILJobStageCondReady)
					klog.Infof("update model for job(%s) successfully", jobConfig.UniqueIdentifier)
				}

				evalModel := job.JobConfig.EvalModel
				if evalModel != nil {
					newEvalModel := im.getModelFromJobConditions(job, sednav1.ILJobEval)
					if err := im.updateDeployModelFile(job, newEvalModel.URL, evalModel.URL); err != nil {
						return err
					}
				}
			} else {
				status.Status = string(sednav1.ILJobStageCondReady)
			}

			status.Input = &clienttypes.Input{
				Models: models,
			}

			klog.Infof("job(%s) completed the %sing phase triggering task successfully",
				jobConfig.UniqueIdentifier, sednav1.ILJobDeploy)
		} else {
			// No need to deploy, just report completed status
			// TODO: instead of reporting deploy-completed, another more reasonable status
			klog.Infof("job(%s) isn't need to deploy model", jobConfig.UniqueIdentifier)
			status.Status = string(sednav1.ILJobStageCondCompleted)
		}

		err = im.Client.WriteMessage(status, job.getHeader())
		if err != nil {
			klog.Errorf("job(%s) completed the %s task failed: %v",
				jobConfig.UniqueIdentifier, sednav1.ILJobDeploy, err)
		}

		job.JobConfig.DeployTriggerStatus = TriggerCompletedStatus
		klog.Infof("job(%s) completed the %s task successfully", jobConfig.UniqueIdentifier, sednav1.ILJobDeploy)
	}

	return nil
}

// startJob starts a job
func (im *Manager) startJob(name string) {
	var err error
	job := im.IncrementalJobMap[name]

	err = im.initJob(job, name)
	if err != nil {
		klog.Errorf("failed to init job (name=%s): %+v", name)
		return
	}

	klog.Infof("incremental job(%s) was started", name)
	defer klog.Infof("incremental learning job(%s) was stopped", name)

	// handle data from dataset
	go im.handleData(job)

	tick := time.NewTicker(JobIterationIntervalSeconds * time.Second)
	for {
		select {
		case <-job.JobConfig.Done:
			return
		default:
		}

		cond := im.getLatestCondition(job)
		jobStage := cond.Stage

		switch jobStage {
		case sednav1.ILJobTrain:
			err = im.trainTask(job)
		case sednav1.ILJobEval:
			err = im.evalTask(job)
		case sednav1.ILJobDeploy:
			if cond.Type == sednav1.ILJobStageCondWaiting {
				err = im.deployTask(job)
			} else if cond.Type == sednav1.ILJobStageCondRunning && job.Spec.DeploySpec.Model.HotUpdateEnabled {
				err = im.hotModelUpdateDeployTask(job)
			}
		default:
			klog.Errorf("invalid phase: %s", jobStage)
			continue
		}

		if err != nil {
			klog.Errorf("job(%s) failed to complete the %s task: %v", name, jobStage, err)
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
		im.IncrementalJobMap[name] = job
		first = true
	}

	if err := json.Unmarshal(message.Content, &job); err != nil {
		return err
	}

	if err := db.SaveResource(name, job.TypeMeta, job.ObjectMeta, job.Spec); err != nil {
		return err
	}

	if first {
		go im.startJob(name)
	}

	return nil
}

// deleteModelHotUpdateData deletes the local data of model hot update
func (im *Manager) deleteModelHotUpdateData(job *Job) error {
	if configFile, ok := job.ObjectMeta.Annotations[runtime.ModelHotUpdateAnnotationsKey]; ok {
		localHostDir := util.AddPrefixPath(im.VolumeMountPrefix, filepath.Dir(configFile))
		dir, err := ioutil.ReadDir(localHostDir)
		if err != nil {
			return err
		}
		for _, d := range dir {
			file := filepath.Join([]string{localHostDir, d.Name()}...)
			if e := os.RemoveAll(file); err != nil {
				return fmt.Errorf("failed to delete the object(%s): %w", file, e)
			}
		}
	}

	return nil
}

// Delete deletes incremental-learning-job config in db
func (im *Manager) Delete(message *clienttypes.Message) error {
	name := util.GetUniqueIdentifier(message.Header.Namespace, message.Header.ResourceName, message.Header.ResourceKind)

	if job, ok := im.IncrementalJobMap[name]; ok && job.JobConfig.Done != nil {
		close(job.JobConfig.Done)

		if err := im.deleteModelHotUpdateData(job); err != nil {
			klog.Errorf("job(%s) failed to delete data of model hot update: %v", name, err)
			// continue anyway
		}
	}

	delete(im.IncrementalJobMap, name)

	if err := db.DeleteResource(name); err != nil {
		return err
	}

	return nil
}

// updateJobFromDB updates job from db
func (im *Manager) updateJobFromDB(job *Job) error {
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
func (im *Manager) saveJobToDB(job *Job) error {
	ann := job.ObjectMeta.Annotations
	if ann == nil {
		ann = make(map[string]string)
	}

	ann[AnnotationsRoundsKey] = strconv.Itoa(job.JobConfig.Rounds)
	ann[AnnotationsNumberOfSamplesKey] = strconv.Itoa(job.JobConfig.DataSamples.PreviousNumbers)
	ann[AnnotationsDataFileOfEvalKey] = job.JobConfig.EvalDataURL

	return db.SaveResource(job.JobConfig.UniqueIdentifier, job.TypeMeta, job.ObjectMeta, job.Spec)
}

// initJob inits the job object
func (im *Manager) initJob(job *Job, name string) error {
	job.JobConfig = new(JobConfig)

	jobConfig := job.JobConfig
	jobConfig.UniqueIdentifier = name

	jobConfig.Storage = storage.Storage{IsLocalStorage: false}
	credential := job.ObjectMeta.Annotations[runtime.SecretAnnotationKey]
	if credential != "" {
		if err := job.JobConfig.Storage.SetCredential(credential); err != nil {
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
	deployTrigger, err := newTrigger(job.Spec.DeploySpec.Trigger)
	if err != nil {
		return fmt.Errorf("failed to init deploy trigger: %+w", err)
	}
	jobConfig.TrainTrigger = trainTrigger
	jobConfig.DeployTrigger = deployTrigger

	outputDir := job.Spec.OutputDir

	isLocalURL, err := jobConfig.Storage.IsLocalURL(outputDir)
	if err != nil {
		return fmt.Errorf("job(%s)'s output dir(%s) is invalid: %+w", job.Name, outputDir, err)
	}

	if isLocalURL {
		jobConfig.Storage.IsLocalStorage = true
		outputDir = util.AddPrefixPath(im.VolumeMountPrefix, outputDir)
	}

	jobConfig.OutputDir = outputDir

	if err := job.createOutputDir(jobConfig); err != nil {
		return err
	}

	if err := im.updateJobFromDB(job); err != nil {
		klog.Errorf("job(%s) failed to update job from db: %v", name, err)
	}

	initTriggerStatus(jobConfig)

	return nil
}

func initTriggerStatus(jobConfig *JobConfig) {
	jobConfig.TrainTriggerStatus = TriggerReadyStatus
	jobConfig.EvalTriggerStatus = TriggerReadyStatus
	jobConfig.DeployTriggerStatus = TriggerReadyStatus
	jobConfig.HotModelUpdateDeployStatus = TriggerReadyStatus
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

// getModelFromJobConditions gets model from job conditions for train/eval/deploy
func (im *Manager) getModelFromJobConditions(job *Job, jobStage sednav1.ILJobStage) *Model {
	jobConditions := job.Status.Conditions
	jobConfig := job.JobConfig

	getModels := func(stage sednav1.ILJobStage, currentType sednav1.ILJobStageConditionType, dataType string) []Model {
		// TODO: runtime.type changes to common.type for gm and lc
		for i := len(jobConditions) - 1; i >= 0; i-- {
			var cond gmtypes.IncrementalCondData
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

	getModel := func(initModel *Model, models []Model) *Model {
		for _, m := range models {
			if m.Format == initModel.Format {
				if initModel.Devices != nil && len(m.Devices) == 1 {
					for _, d := range initModel.Devices {
						if m.Devices[0] == d {
							return &m
						}
					}
				} else {
					return &m
				}
			}
		}

		return nil
	}

	models := getModels(sednav1.ILJobTrain, sednav1.ILJobStageCondCompleted, "output")
	if models == nil {
		return nil
	}

	var model *Model
	switch jobStage {
	case sednav1.ILJobTrain:
		model = jobConfig.TrainModel
	case sednav1.ILJobEval:
		model = jobConfig.EvalModel
	case sednav1.ILJobDeploy:
		model = jobConfig.DeployModel
	}
	if model == nil {
		return nil
	}

	return getModel(model, models)
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

	job.JobConfig.Rounds++

	var m *Model
	rounds := jobConfig.Rounds
	if rounds <= 1 {
		m = jobConfig.TrainModel
	} else {
		m = im.getModelFromJobConditions(job, sednav1.ILJobTrain)
	}

	var dataIndexURL string
	jobConfig.TrainDataURL, dataIndexURL, err = im.writeSamples(job, jobConfig.DataSamples.TrainSamples,
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

	m := im.getModelFromJobConditions(job, sednav1.ILJobEval)

	var models []Model
	models = append(models, *m, *jobConfig.EvalModel)

	var dataIndexURL string
	jobConfig.EvalDataURL, dataIndexURL, err = im.writeSamples(job, jobConfig.DataSamples.EvalSamples, jobConfig.OutputConfig.SamplesOutput["eval"],
		job.JobConfig.Rounds, jobConfig.Dataset.Spec.Format, jobConfig.Dataset.URLPrefix)
	if err != nil {
		klog.Errorf("job(%s) eval phase: write samples to the file(%s) is failed: %v",
			jobConfig.UniqueIdentifier, jobConfig.EvalDataURL, err)
		return nil, err
	}
	jobConfig.DataSamples.EvalSamples = []string{}

	dataURL := jobConfig.EvalDataURL
	if jobConfig.Storage.IsLocalStorage {
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

// updateDeployModelFile updates deploy model file
func (im *Manager) updateDeployModelFile(job *Job, trainedModel string, deployModel string) error {
	if job.JobConfig.Storage.IsLocalStorage {
		trainedModel = util.AddPrefixPath(im.VolumeMountPrefix, trainedModel)
	}

	if err := job.JobConfig.Storage.CopyFile(trainedModel, deployModel); err != nil {
		return fmt.Errorf("failed to copy trained model(url=%s) to the deploy model(url=%s): %w",
			trainedModel, deployModel, err)
	}

	klog.V(4).Infof("copy trained model(url=%s) to the deploy model(url=%s) successfully", trainedModel, deployModel)

	return nil
}

// createOutputDir creates the job output dir
func (job *Job) createOutputDir(jobConfig *JobConfig) error {
	outputDir := jobConfig.OutputDir

	dirNames := []string{"data/train", "data/eval", "train", "eval"}

	if job.JobConfig.Storage.IsLocalStorage {
		if err := util.CreateFolder(outputDir); err != nil {
			klog.Errorf("job(%s) failed to create folder %s: %v", jobConfig.UniqueIdentifier, outputDir, err)
			return err
		}

		for _, v := range dirNames {
			dir := path.Join(outputDir, v)
			if err := util.CreateFolder(dir); err != nil {
				klog.Errorf("job(%s) failed to create folder %s: %v", jobConfig.UniqueIdentifier, dir, err)
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
		jobConfig.TrainModel.Format = initialModel.Spec.Format
		jobConfig.TrainModel.URL = initialModel.Spec.URL
		jobConfig.TrainModel.Devices = initialModel.Spec.Devices
	}
	return nil
}

// loadEvalModel loads initial model information for eval.
func (im *Manager) loadEvalModel(job *Job) error {
	jobConfig := job.JobConfig

	if jobConfig.EvalModel == nil {
		initialModel, err := im.getModel(job.Namespace, job.Spec.EvalSpec.InitialModel.Name)
		if err != nil {
			return err
		}

		jobConfig.EvalModel = new(Model)
		jobConfig.EvalModel.Format = initialModel.Spec.Format
		jobConfig.EvalModel.URL = initialModel.Spec.URL
		jobConfig.EvalModel.Devices = initialModel.Spec.Devices
	}
	return nil
}

// loadDeployModel loads model information for deploying.
func (im *Manager) loadDeployModel(job *Job) error {
	jobConfig := job.JobConfig

	if jobConfig.DeployModel == nil {
		deployModel, err := im.getModel(job.Namespace, job.Spec.DeploySpec.Model.Name)
		if err != nil {
			return err
		}

		jobConfig.DeployModel = new(Model)
		jobConfig.DeployModel.Format = deployModel.Spec.Format
		jobConfig.DeployModel.URL = deployModel.Spec.URL
		jobConfig.DeployModel.Devices = deployModel.Spec.Devices
	}
	return nil
}

// loadDataset loads dataset information
func (im *Manager) loadDataset(job *Job) error {
	if job.JobConfig.Dataset != nil {
		// already loaded
		return nil
	}

	datasetName := util.GetUniqueIdentifier(job.Namespace, job.Spec.Dataset.Name, dataset.KindName)
	dataset, ok := im.DatasetManager.GetDataset(datasetName)
	if !ok || dataset == nil {
		return fmt.Errorf("not exists dataset(name=%s)", datasetName)
	}

	job.JobConfig.Dataset = dataset
	return nil
}

// handleData updates samples information
func (im *Manager) handleData(job *Job) {
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
				samples[(previousNumberOfSamples):(previousNumberOfSamples+trainNum)]...)
			klog.Infof("job(%s)'s current train samples nums is %d", jobConfig.UniqueIdentifier, trainNum)

			jobConfig.DataSamples.EvalVersionSamples = append(jobConfig.DataSamples.EvalVersionSamples,
				samples[(previousNumberOfSamples+trainNum):])
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
		if err := im.writeByLine(samples, fileURL); err != nil {
			return "", "", err
		}

		if !jobConfig.Dataset.Storage.IsLocalStorage {
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

	localFileURL, localAbsURLFile := createFile(temporaryDir, format, jobConfig.Dataset.Storage.IsLocalStorage)

	if err := im.writeByLine(samples, localFileURL); err != nil {
		return "", "", err
	}

	if err := jobConfig.Storage.Upload(localFileURL, fileURL); err != nil {
		return "", "", err
	}

	tempSamples := util.ParsingDatasetIndex(samples, urlPrefix)

	if err := im.writeByLine(tempSamples, localAbsURLFile); err != nil {
		return "", "", err
	}

	if err := jobConfig.Storage.Upload(localAbsURLFile, absURLFile); err != nil {
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
		klog.Errorf("failed to write file(%s): %v", fileURL, err)
		return err
	}

	if err := file.Close(); err != nil {
		klog.Errorf("failed to close file(%s): %v", fileURL, err)
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
			klog.Errorf("job(%s) failed to write message: %v", name, err)
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
		klog.Warningf("job(%s)'s %s phase get worker(kind=%s)", job.JobConfig.UniqueIdentifier,
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
		klog.Infof("job(%s) completed the %s task successfully", jobName, jobStage)
		switch latestCond.Stage {
		case sednav1.ILJobEval:
			job.JobConfig.EvalResult = models
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
