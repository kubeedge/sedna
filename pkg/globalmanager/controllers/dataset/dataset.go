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
	sednaclientset "github.com/kubeedge/sedna/pkg/client/clientset/versioned/typed/sedna/v1alpha1"
	"github.com/kubeedge/sedna/pkg/globalmanager/config"

	"github.com/kubeedge/sedna/pkg/globalmanager/runtime"
)

const (
	// KindName is the kind name of CR this controller controls
	KindName = "Dataset"

	// Name is this controller name
	Name = "Dataset"
)

// Controller handles all dataset objects including: syncing to edge and update from edge.
type Controller struct {
	client sednaclientset.SednaV1alpha1Interface

	cfg *config.ControllerConfig
}

func (c *Controller) Run(stopCh <-chan struct{}) {
	// noop now
}

// New creates a dataset controller
func New(cc *runtime.ControllerContext) (runtime.FeatureControllerI, error) {
	c := &Controller{
		client: cc.SednaClient.SednaV1alpha1(),
	}

	c.addUpstreamHandler(cc)

	return c, nil
}
