package manager

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"k8s.io/klog/v2"

	"github.com/edgeai-neptune/neptune/cmd/neptune-lc/app/options"
	neptunev1 "github.com/edgeai-neptune/neptune/pkg/apis/neptune/v1alpha1"
	"github.com/edgeai-neptune/neptune/pkg/localcontroller/db"
	"github.com/edgeai-neptune/neptune/pkg/localcontroller/gmclient"
	"github.com/edgeai-neptune/neptune/pkg/localcontroller/util"
)

const (
	// MonitorDataSourceIntervalSeconds is interval time of monitoring data source
	MonitorDataSourceIntervalSeconds = 10
	// DatasetResourceKind is kind of dataset resource
	DatasetResourceKind = "dataset"
)

// DatasetManager defines dataset manager
type DatasetManager struct {
	Client            gmclient.ClientI
	DatasetMap        map[string]*Dataset
	VolumeMountPrefix string
}

// Dataset defines config for dataset
type Dataset struct {
	*neptunev1.Dataset
	DataSource *DataSource `json:"dataSource"`
	Done       chan struct{}
}

// DatasetSpec defines dataset spec
type DatasetSpec struct {
	Format  string `json:"format"`
	DataURL string `json:"url"`
}

// DataSource defines config for data source
type DataSource struct {
	TrainSamples    []string `json:"trainSamples"`
	ValidSamples    []string `json:"validSamples"`
	NumberOfSamples int      `json:"numberOfSamples"`
}

// NewDatasetManager creates a dataset manager
func NewDatasetManager(client gmclient.ClientI, options *options.LocalControllerOptions) *DatasetManager {
	dm := DatasetManager{
		Client:            client,
		DatasetMap:        make(map[string]*Dataset),
		VolumeMountPrefix: options.VolumeMountPrefix,
	}

	return &dm
}

// Start starts dataset manager
func (dm *DatasetManager) Start() error {
	return nil
}

// GetDatasetChannel gets dataset
func (dm *DatasetManager) GetDataset(name string) (*Dataset, bool) {
	d, ok := dm.DatasetMap[name]
	return d, ok
}

// Insert inserts dataset to db
func (dm *DatasetManager) Insert(message *gmclient.Message) error {
	name := util.GetUniqueIdentifier(message.Header.Namespace, message.Header.ResourceName, message.Header.ResourceKind)
	first := false
	dataset, ok := dm.DatasetMap[name]
	if !ok {
		dataset = &Dataset{}
		dataset.Done = make(chan struct{})
		dm.DatasetMap[name] = dataset
		first = true
	}

	if err := json.Unmarshal(message.Content, dataset); err != nil {
		return err
	}

	if first {
		go dm.monitorDataSources(name)
	}

	if err := db.SaveResource(name, dataset.TypeMeta, dataset.ObjectMeta, dataset.Spec); err != nil {
		return err
	}

	return nil
}

// Delete deletes dataset config in db
func (dm *DatasetManager) Delete(message *gmclient.Message) error {
	name := util.GetUniqueIdentifier(message.Header.Namespace, message.Header.ResourceName, message.Header.ResourceKind)

	if ds, ok := dm.DatasetMap[name]; ok && ds.Done != nil {
		close(ds.Done)
	}

	delete(dm.DatasetMap, name)

	if err := db.DeleteResource(name); err != nil {
		return err
	}

	return nil
}

// monitorDataSources monitors the data url of specified dataset
func (dm *DatasetManager) monitorDataSources(name string) {
	ds, ok := dm.DatasetMap[name]
	if !ok || ds == nil {
		return
	}

	if ds.Spec.URL == "" {
		klog.Errorf("dataset(name=%s) empty data source url.", name)
		return
	}

	for {
		select {
		case <-ds.Done:
			return
		default:
		}

		dataURL := util.AddPrefixPath(dm.VolumeMountPrefix, filepath.Join(ds.Spec.URL))
		dataSource, err := dm.getDataSource(dataURL, ds.Spec.Format)
		if err != nil {
			klog.Errorf("dataset(name=%s) get samples from %s failed", name, dataURL)
		} else {
			ds.DataSource = dataSource

			klog.Infof("dataset(name=%s) get samples from data source(url=%s) successfully. number of samples: %d",
				name, dataURL, dataSource.NumberOfSamples)

			header := gmclient.MessageHeader{
				Namespace:    ds.Namespace,
				ResourceKind: ds.Kind,
				ResourceName: ds.Name,
				Operation:    gmclient.StatusOperation,
			}

			if err := dm.Client.WriteMessage(struct {
				NumberOfSamples int `json:"numberOfSamples"`
			}{
				dataSource.NumberOfSamples,
			}, header); err != nil {
				klog.Errorf("dataset(name=%s) publish samples info failed", name)
			}
		}
		<-time.After(MonitorDataSourceIntervalSeconds * time.Second)
	}
}

// getDataSource gets data source info
func (dm *DatasetManager) getDataSource(dataURL string, format string) (*DataSource, error) {
	switch format {
	case "txt":
		return dm.readByLine(dataURL)
	}
	return nil, fmt.Errorf("not vaild file format")
}

// readByLine reads file by line
func (dm *DatasetManager) readByLine(url string) (*DataSource, error) {
	samples, err := getSamples(url)
	if err != nil {
		klog.Errorf("read file %s failed, error: %v", url, err)
		return nil, err
	}

	numberOfSamples := 0
	numberOfSamples += len(samples)

	dataSource := DataSource{
		TrainSamples:    samples,
		NumberOfSamples: numberOfSamples,
	}

	return &dataSource, nil
}

func (dm *DatasetManager) GetName() string {
	return DatasetResourceKind
}

func (dm *DatasetManager) AddWorkerMessage(message WorkerMessage) {
	// dummy
}

// getSamples gets samples in a file
func getSamples(url string) ([]string, error) {
	var samples = []string{}
	if !util.IsExists(url) {
		return nil, fmt.Errorf("url(%s) does not exist", url)
	}

	if !util.IsFile(url) {
		return nil, fmt.Errorf("url(%s) is not a file, not vaild", url)
	}

	file, err := os.Open(url)
	if err != nil {
		klog.Errorf("read %s failed, error: %v", url, err)
		return samples, err
	}

	fileScanner := bufio.NewScanner(file)
	for fileScanner.Scan() {
		samples = append(samples, fileScanner.Text())
	}

	if err = file.Close(); err != nil {
		klog.Errorf("close file(url=%s) failed, error: %v", url, err)
		return samples, err
	}

	return samples, nil
}
