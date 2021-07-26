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

// CommonInterface describes the commom interface of CRs
type CommonInterface interface {
	metav1.Object
	schema.ObjectKind
	k8sruntime.Object
}

// BaseControllerI defines the interface of an controller
type BaseControllerI interface {
	Run(stopCh <-chan struct{})
}

// FeatureControllerI defines the interface of an AI Feature controller
type FeatureControllerI interface {
	BaseControllerI
	SetDownstreamSendFunc(f DownstreamSendFunc) error
}

type Model struct {
	Format  string                 `json:"format,omitempty"`
	URL     string                 `json:"url,omitempty"`
	Metrics map[string]interface{} `json:"metrics,omitempty"`
}

const (
	// TrainPodType is type of train pod
	TrainPodType = "train"
	// EvalPodType is type of eval pod
	EvalPodType = "eval"
	// InferencePodType is type of inference pod
	InferencePodType = "inference"

	// AnnotationsKeyPrefix defines prefix of key in annotations
	AnnotationsKeyPrefix = "sedna.io/"
)

func (m *Model) GetURL() string {
	return m.URL
}

// updateHandler handles the updates from LC(running at edge) to update the
// corresponding resource
type UpstreamUpdateHandler func(namespace, name, operation string, content []byte) error

type UpstreamControllerI interface {
	BaseControllerI
	Add(kind string, updateHandler UpstreamUpdateHandler) error
}

type DownstreamSendFunc = func(nodeName string, eventType watch.EventType, obj interface{}) error

type ControllerContext struct {
	Config             *config.ControllerConfig
	UpstreamController UpstreamControllerI

	KubeClient          kubernetes.Interface
	KubeInformerFactory kubeinformers.SharedInformerFactory

	SednaClient          sednaclientset.Interface
	SednaInformerFactory sednainformers.SharedInformerFactory
}
