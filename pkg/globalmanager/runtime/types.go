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

package runtime

import (
	"time"

	"github.com/kubeedge/sedna/pkg/globalmanager/config"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8sruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/watch"
	kubeinformers "k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"

	sednaclientset "github.com/kubeedge/sedna/pkg/client/clientset/versioned"
	sednainformers "github.com/kubeedge/sedna/pkg/client/informers/externalversions"
)

const (
	// DefaultBackOff is the default backoff period
	DefaultBackOff = 10 * time.Second
	// MaxBackOff is the max backoff period
	MaxBackOff = 360 * time.Second

	// TrainPodType is type of train pod
	TrainPodType = "train"
	// EvalPodType is type of eval pod
	EvalPodType = "eval"
	// InferencePodType is type of inference pod
	InferencePodType = "inference"

	// AnnotationsKeyPrefix defines prefix of key in annotations
	AnnotationsKeyPrefix = "sedna.io/"

	ModelHotUpdateHostPrefix      = "/var/lib/sedna/model-hot-update"
	ModelHotUpdateContainerPrefix = "/model-hot-update"
	ModelHotUpdateVolumeName      = "sedna-model-hot-update-volume"
	ModelHotUpdateConfigFile      = "model_config.json"
	ModelHotUpdateAnnotationsKey  = "sedna.io/model-hot-update-config"
)

type Model struct {
	Format        string                 `json:"format"`
	URL           string                 `json:"url"`
	Devices       []string               `json:"device_soc_versions,omitempty"`
	Metrics       map[string]interface{} `json:"metrics,omitempty"`
	HistoryMetric float32                `json:"history_metric,omitempty"`
	CurrentMetric float32                `json:"current_metric,omitempty"`

	//TODO:这些应该被挪到Model对象外面。
	NumberOfLabeledUnseenSample int `json:"number_of_labeled_unseen_sample,omitempty"`
	NumberOfModel               int `json:"number_of_model,omitempty"`
	NumberOfUnseenSample        int `json:"number_of_unseen_sample,omitempty"`
}

func (m *Model) GetURL() string {
	return m.URL
}

// CommonInterface describes the commom interface of CRs
type CommonInterface interface {
	metav1.Object
	schema.ObjectKind
	k8sruntime.Object
}

// UpstreamHandler is the function definition for handling the upstream updates,
// i.e. resource updates(mainly status) from LC(running at edge)
type UpstreamHandler = func(namespace, name, operation string, content []byte) error

// UpstreamHandlerAddFunc defines the upstream controller register function for adding handler
type UpstreamHandlerAddFunc = func(kind string, updateHandler UpstreamHandler) error

// DownstreamSendFunc is the send function for feature controllers to sync the resource updates(spec and status) to LC
type DownstreamSendFunc = func(nodeName string, eventType watch.EventType, obj interface{}) error

// BaseControllerI defines the interface of an controller
type BaseControllerI interface {
	Run(stopCh <-chan struct{})
}

// FeatureControllerI defines the interface of an AI Feature controller
type FeatureControllerI interface {
	BaseControllerI

	// SetDownstreamSendFunc sets up the downstream send function in the feature controller
	SetDownstreamSendFunc(f DownstreamSendFunc) error

	// SetUpstreamHandler sets up the upstream handler function for the feature controller
	SetUpstreamHandler(add UpstreamHandlerAddFunc) error
}

// ControllerContext defines the context that all feature controller share and belong to
type ControllerContext struct {
	Config *config.ControllerConfig

	KubeClient          kubernetes.Interface
	KubeInformerFactory kubeinformers.SharedInformerFactory

	SednaClient          sednaclientset.Interface
	SednaInformerFactory sednainformers.SharedInformerFactory
}
