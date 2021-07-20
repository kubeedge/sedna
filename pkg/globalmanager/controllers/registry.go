package controllers

import (
	"fmt"

	"github.com/kubeedge/sedna/pkg/globalmanager/config"
	fl "github.com/kubeedge/sedna/pkg/globalmanager/controllers/federatedlearning"
	il "github.com/kubeedge/sedna/pkg/globalmanager/controllers/incrementallearning"
	ji "github.com/kubeedge/sedna/pkg/globalmanager/controllers/jointinference"
	ll "github.com/kubeedge/sedna/pkg/globalmanager/controllers/lifelonglearning"
	"github.com/kubeedge/sedna/pkg/globalmanager/runtime"
)

type FeatureFactory = func(cfg *config.ControllerConfig) (runtime.FeatureControllerI, error)

type Registry map[string]FeatureFactory

func (r Registry) Register(name string, factory FeatureFactory) error {
	if _, ok := r[name]; ok {
		return fmt.Errorf("a feature controller named %s already exists", name)
	}
	r[name] = factory
	return nil
}

func NewRegistry() Registry {
	return Registry{
		ji.Name: ji.New,
		fl.Name: fl.New,
		il.Name: il.New,
		ll.Name: ll.New,
	}
}
