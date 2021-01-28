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

package options

import (
	"fmt"
	"path"

	"k8s.io/apimachinery/pkg/util/validation/field"
	cliflag "k8s.io/component-base/cli/flag"

	"github.com/kubeedge/sedna/pkg/globalmanager/config"
	"github.com/kubeedge/sedna/pkg/util"
)

// DefaultConfigDir default current working directory
const DefaultConfigDir = "."

// ControllerOptions describes gm options
type ControllerOptions struct {
	ConfigFile string
}

// NewControllerOptions creates a new gm options
func NewControllerOptions() *ControllerOptions {
	return &ControllerOptions{
		ConfigFile: path.Join(DefaultConfigDir, "sedna-gm.yaml"),
	}
}

// Flags returns flags of ControllerOptions
func (o *ControllerOptions) Flags() (fss cliflag.NamedFlagSets) {
	fs := fss.FlagSet("global")
	fs.StringVar(&o.ConfigFile, "config", o.ConfigFile, "The path to the configuration file. Flags override values in this file.")
	return
}

// Validate validates the ControllerOptions
func (o *ControllerOptions) Validate() []error {
	var errs []error
	if !util.FileIsExist(o.ConfigFile) {
		errs = append(errs, field.Required(field.NewPath("config"),
			fmt.Sprintf("config file %v not exist.", o.ConfigFile)))
	}
	return errs
}

// Config returns a config.ControllerConfig
func (o *ControllerOptions) Config() (*config.ControllerConfig, error) {
	cfg := config.NewDefaultControllerConfig()
	if err := cfg.Parse(o.ConfigFile); err != nil {
		return nil, err
	}
	return cfg, nil
}
