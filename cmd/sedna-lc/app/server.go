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

package app

import (
	"fmt"
	"os"
	"path"

	"github.com/spf13/cobra"
	cliflag "k8s.io/component-base/cli/flag"
	"k8s.io/component-base/cli/globalflag"
	"k8s.io/klog/v2"

	"github.com/kubeedge/sedna/cmd/sedna-lc/app/options"
	"github.com/kubeedge/sedna/pkg/localcontroller/common/constants"
	"github.com/kubeedge/sedna/pkg/localcontroller/gmclient"
	"github.com/kubeedge/sedna/pkg/localcontroller/managers"
	"github.com/kubeedge/sedna/pkg/localcontroller/managers/dataset"
	"github.com/kubeedge/sedna/pkg/localcontroller/managers/federatedlearning"
	"github.com/kubeedge/sedna/pkg/localcontroller/managers/incrementallearning"
	"github.com/kubeedge/sedna/pkg/localcontroller/managers/jointinference"
	"github.com/kubeedge/sedna/pkg/localcontroller/managers/lifelonglearning"
	"github.com/kubeedge/sedna/pkg/localcontroller/managers/model"
	"github.com/kubeedge/sedna/pkg/localcontroller/managers/multiedgetracking"
	"github.com/kubeedge/sedna/pkg/localcontroller/server"
	"github.com/kubeedge/sedna/pkg/version/verflag"
)

var (
	// Options defines the lc options
	Options *options.LocalControllerOptions
)

// NewLocalControllerCommand creates a command object
func NewLocalControllerCommand() *cobra.Command {
	cmdName := path.Base(os.Args[0])
	cmd := &cobra.Command{
		Use: cmdName,
		Long: fmt.Sprintf(`%s is the localcontroller.
It manages dataset and models, and controls ai features in local nodes.`, cmdName),
		Run: func(cmd *cobra.Command, args []string) {
			runServer()
		},
	}

	fs := cmd.Flags()
	namedFs := cliflag.NamedFlagSets{}

	verflag.AddFlags(namedFs.FlagSet("global"))
	globalflag.AddGlobalFlags(namedFs.FlagSet("global"), cmd.Name())
	for _, f := range namedFs.FlagSets {
		fs.AddFlagSet(f)
	}

	Options = options.NewLocalControllerOptions()

	Options.GMAddr = os.Getenv(constants.GMAddressENV)
	if Options.NodeName = os.Getenv(constants.NodeNameENV); Options.NodeName == "" {
		Options.NodeName = os.Getenv(constants.HostNameENV)
	}

	var ok bool
	if Options.VolumeMountPrefix, ok = os.LookupEnv(constants.RootFSMountDirENV); !ok {
		Options.VolumeMountPrefix = "/rootfs"
	}

	if Options.BindPort = os.Getenv(constants.BindPortENV); Options.BindPort == "" {
		Options.BindPort = "9100"
	}

	return cmd
}

// runServer runs server
func runServer() {
	c := gmclient.NewWebSocketClient(Options)
	if err := c.Start(); err != nil {
		return
	}

	dm := dataset.New(c, Options)

	mm := model.New(c)

	jm := jointinference.New(c)

	me := multiedgetracking.New(c)

	fm := federatedlearning.New(c)

	im := incrementallearning.New(c, dm, mm, Options)

	lm := lifelonglearning.New(c, dm, Options)

	s := server.New(Options)

	for _, m := range []managers.FeatureManager{
		dm, me, mm, jm, fm, im, lm,
	} {
		s.AddFeatureManager(m)
		c.Subscribe(m)
		err := m.Start()
		if err != nil {
			klog.Errorf("failed to start manager %s: %v",
				m.GetName(), err)
			return
		}
		klog.Infof("manager %s is started", m.GetName())
	}

	s.ListenAndServe()
}
