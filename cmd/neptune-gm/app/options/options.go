package options

import (
	"fmt"
	"path"

	"k8s.io/apimachinery/pkg/util/validation/field"
	cliflag "k8s.io/component-base/cli/flag"

	"github.com/edgeai-neptune/neptune/pkg/globalmanager/config"
	"github.com/edgeai-neptune/neptune/pkg/util"
)

const DefaultConfigDir = "."

type ControllerOptions struct {
	ConfigFile string
}

func NewControllerOptions() *ControllerOptions {
	return &ControllerOptions{
		ConfigFile: path.Join(DefaultConfigDir, "neptune-gm.yaml"),
	}
}

func (o *ControllerOptions) Flags() (fss cliflag.NamedFlagSets) {
	fs := fss.FlagSet("global")
	fs.StringVar(&o.ConfigFile, "config", o.ConfigFile, "The path to the configuration file. Flags override values in this file.")
	return
}

func (o *ControllerOptions) Validate() []error {
	var errs []error
	if !util.FileIsExist(o.ConfigFile) {
		errs = append(errs, field.Required(field.NewPath("config"),
			fmt.Sprintf("config file %v not exist.", o.ConfigFile)))
	}
	return errs
}

func (o *ControllerOptions) Config() (*config.ControllerConfig, error) {
	cfg := config.NewDefaultControllerConfig()
	if err := cfg.Parse(o.ConfigFile); err != nil {
		return nil, err
	}
	return cfg, nil
}
