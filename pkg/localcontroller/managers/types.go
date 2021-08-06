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

package managers

import (
	clienttype "github.com/kubeedge/sedna/pkg/localcontroller/gmclient"
	workertypes "github.com/kubeedge/sedna/pkg/localcontroller/worker"
)

// FeatureManager defines feature managers
type FeatureManager interface {
	// Start starts the managers
	Start() error

	// GetName returns name of the managers
	GetName() string

	// AddWorkerMessage dispatch the worker message to managers
	AddWorkerMessage(message workertypes.MessageContent)

	// Insert includes gm message creation/updation
	Insert(*clienttype.Message) error

	Delete(*clienttype.Message) error
}
