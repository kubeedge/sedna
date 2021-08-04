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

package dataset

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"k8s.io/klog/v2"

	"github.com/kubeedge/sedna/cmd/sedna-lc/app/options"
	sednav1 "github.com/kubeedge/sedna/pkg/apis/sedna/v1alpha1"
	"github.com/kubeedge/sedna/pkg/globalmanager/runtime"
	"github.com/kubeedge/sedna/pkg/localcontroller/db"
	clienttypes "github.com/kubeedge/sedna/pkg/localcontroller/gmclient"
	"github.com/kubeedge/sedna/pkg/localcontroller/storage"
	"github.com/kubeedge/sedna/pkg/localcontroller/util"
	workertypes "github.com/kubeedge/sedna/pkg/localcontroller/worker"
)

const (
	// MonitorDataSourceIntervalSeconds is interval time of monitoring data source
	MonitorDataSourceIntervalSeconds = 60
	// KindName is kind of dataset resource
	KindName = "dataset"
	// CSVFormat is commas separated value format with a extra header.
	// It can be used in structured data scenarios.
	CSVFormat = "csv"
	// FormatTXT is line separated format.
	// It can be used in unstructured data scenarios.
	TXTFormat = "txt"
)

// DatasetManager defines dataset manager
type Manager struct {
	Client            clienttypes.ClientI
	DatasetMap        map[string]*Dataset
	VolumeMountPrefix string
}

// Dataset defines config for dataset
type Dataset struct {
	*sednav1.Dataset
	DataSource *DataSource `json:"dataSource"`
	Done       chan struct{}
	URLPrefix  string
	Storage    storage.Storage
}

// DataSource defines config for data source
type DataSource struct {
	TrainSamples    []string
	NumberOfSamples int
	Header          string
}

// New creates a dataset manager
func New(client clienttypes.ClientI, options *options.LocalControllerOptions) *Manager {
	dm := Manager{
		Client:            client,
		DatasetMap:        make(map[string]*Dataset),
		VolumeMountPrefix: options.VolumeMountPrefix,
	}

	return &dm
}

// Start starts dataset manager
func (dm *Manager) Start() error {
	return nil
}

// GetDatasetChannel gets dataset
func (dm *Manager) GetDataset(name string) (*Dataset, bool) {
	d, ok := dm.DatasetMap[name]
	return d, ok
}

// Insert inserts dataset to db
func (dm *Manager) Insert(message *clienttypes.Message) error {
	name := util.GetUniqueIdentifier(message.Header.Namespace, message.Header.ResourceName, message.Header.ResourceKind)
	first := false
	dataset, ok := dm.DatasetMap[name]
	if !ok {
		dataset = &Dataset{}
		dataset.Storage = storage.Storage{IsLocalStorage: false}
		dataset.Done = make(chan struct{})
		dm.DatasetMap[name] = dataset
		first = true
	}

	if err := json.Unmarshal(message.Content, dataset); err != nil {
		return err
	}

	credential := dataset.ObjectMeta.Annotations[runtime.SecretAnnotationKey]
	if credential != "" {
		if err := dataset.Storage.SetCredential(credential); err != nil {
			return fmt.Errorf("failed to set dataset(name=%s)'s storage credential, error: %+v", name, err)
		}
	}

	isLocalURL, err := dataset.Storage.IsLocalURL(dataset.Spec.URL)
	if err != nil {
		return fmt.Errorf("dataset(name=%s)'s url is invalid, error: %+v", name, err)
	}
	if isLocalURL {
		dataset.Storage.IsLocalStorage = true
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
func (dm *Manager) Delete(message *clienttypes.Message) error {
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
func (dm *Manager) monitorDataSources(name string) {
	ds, ok := dm.DatasetMap[name]
	if !ok || ds == nil {
		return
	}

	dataURL := ds.Spec.URL
	if ds.Storage.IsLocalStorage {
		dataURL = util.AddPrefixPath(dm.VolumeMountPrefix, dataURL)
	}

	ds.URLPrefix = strings.TrimRight(dataURL, filepath.Base(dataURL))
	samplesNumber := 0
	for {
		select {
		case <-ds.Done:
			return
		default:
		}

		dataSource, err := ds.getDataSource(dataURL, ds.Spec.Format)
		if err != nil {
			klog.Errorf("dataset(name=%s) get samples from %s failed, error: %+v", name, dataURL, err)
		} else {
			ds.DataSource = dataSource
			if samplesNumber != dataSource.NumberOfSamples {
				samplesNumber = dataSource.NumberOfSamples
				klog.Infof("dataset(name=%s) get samples from data source(url=%s) successfully. number of samples: %d",
					name, dataURL, dataSource.NumberOfSamples)

				header := clienttypes.MessageHeader{
					Namespace:    ds.Namespace,
					ResourceKind: ds.Kind,
					ResourceName: ds.Name,
					Operation:    clienttypes.StatusOperation,
				}

				if err := dm.Client.WriteMessage(struct {
					NumberOfSamples int `json:"numberOfSamples"`
				}{
					dataSource.NumberOfSamples,
				}, header); err != nil {
					klog.Errorf("dataset(name=%s) publish samples info failed, error: %+v", name, err)
				}
			}
		}
		<-time.After(MonitorDataSourceIntervalSeconds * time.Second)
	}
}

// getDataSource gets data source info
func (ds *Dataset) getDataSource(dataURL string, format string) (*DataSource, error) {
	if err := ds.validFormat(format); err != nil {
		return nil, err
	}

	localURL, err := ds.Storage.Download(dataURL, "")

	if !ds.Storage.IsLocalStorage {
		defer os.RemoveAll(localURL)
	}

	if err != nil {
		return nil, err
	}

	return ds.readByLine(localURL, format)
}

// readByLine reads file by line
func (ds *Dataset) readByLine(url string, format string) (*DataSource, error) {
	samples, err := GetSamples(url)
	if err != nil {
		klog.Errorf("read file %s failed, error: %v", url, err)
		return nil, err
	}

	numberOfSamples := 0
	dataSource := DataSource{}
	switch strings.ToLower(format) {
	case TXTFormat:
		numberOfSamples += len(samples)
	case CSVFormat:
		// the first row of csv file is header
		if len(samples) == 0 {
			return nil, fmt.Errorf("file %s is empty", url)
		}
		dataSource.Header = samples[0]
		samples = samples[1:]
		numberOfSamples += len(samples)

	default:
		return nil, fmt.Errorf("invaild file format")
	}

	dataSource.TrainSamples = samples
	dataSource.NumberOfSamples = numberOfSamples

	return &dataSource, nil
}

func (dm *Manager) GetName() string {
	return KindName
}

func (dm *Manager) AddWorkerMessage(message workertypes.MessageContent) {
	// dummy
}

// GetSamples gets samples in a file
func GetSamples(url string) ([]string, error) {
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

// validFormat checks data format is valid
func (ds *Dataset) validFormat(format string) error {
	for _, v := range []string{TXTFormat, CSVFormat} {
		if strings.ToLower(format) == v {
			return nil
		}
	}

	return fmt.Errorf("dataset format(%s) is invalid", format)
}
