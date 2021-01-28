package app

import (
	"fmt"
	"os"
	"path"

	"github.com/spf13/cobra"
	cliflag "k8s.io/component-base/cli/flag"
	"k8s.io/component-base/cli/globalflag"
	"k8s.io/klog/v2"

	"github.com/edgeai-neptune/neptune/cmd/neptune-lc/app/options"
	"github.com/edgeai-neptune/neptune/pkg/localcontroller/common/constants"
	"github.com/edgeai-neptune/neptune/pkg/localcontroller/gmclient"
	"github.com/edgeai-neptune/neptune/pkg/localcontroller/manager"
	"github.com/edgeai-neptune/neptune/pkg/localcontroller/server"
	"github.com/edgeai-neptune/neptune/pkg/version/verflag"
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

	if Options.VolumeMountPrefix = os.Getenv(constants.RootFSMountDirENV); Options.VolumeMountPrefix == "" {
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

	dm := manager.NewDatasetManager(c, Options)

	mm := manager.NewModelManager(c)

	jm := manager.NewJointInferenceManager(c)

	fm := manager.NewFederatedLearningManager(c)

	im := manager.NewIncrementalJobManager(c, dm, mm, Options)

	s := server.New(Options)

	for _, m := range []manager.FeatureManager{
		dm, mm, jm, fm, im,
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
