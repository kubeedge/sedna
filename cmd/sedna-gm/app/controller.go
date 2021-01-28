package app

import (
	"fmt"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
	cliflag "k8s.io/component-base/cli/flag"
	"k8s.io/component-base/cli/globalflag"
	"k8s.io/component-base/term"
	"k8s.io/klog/v2"

	"github.com/kubeedge/sedna/cmd/sedna-gm/app/options"
	controller "github.com/kubeedge/sedna/pkg/globalmanager"
	"github.com/kubeedge/sedna/pkg/util"
	"github.com/kubeedge/sedna/pkg/version/verflag"
)

// PrintFlags logs the flags in the flagset
func PrintFlags(flags *pflag.FlagSet) {
	flags.VisitAll(func(flag *pflag.Flag) {
		klog.V(1).Infof("FLAG: --%s=%q", flag.Name, flag.Value)
	})
}

// NewControllerCommand creates a new gm command
func NewControllerCommand() *cobra.Command {
	opts := options.NewControllerOptions()
	cmd := &cobra.Command{
		Use:  "sedna-gm",
		Long: `sedna-gm is the core cloud part of Sedna.`,
		Run: func(cmd *cobra.Command, args []string) {
			verflag.PrintAndExitIfRequested()
			PrintFlags(cmd.Flags())

			if errs := opts.Validate(); len(errs) > 0 {
				klog.Fatal(util.SpliceErrors(errs))
			}

			config, err := opts.Config()
			if err != nil {
				klog.Fatal(err)
			}

			if errs := config.Validate(); len(errs) > 0 {
				klog.Fatal(util.SpliceErrors(errs.ToAggregate().Errors()))
			}
			c := controller.NewController(config)
			c.Start()
		},
	}
	fs := cmd.Flags()
	namedFs := opts.Flags()
	verflag.AddFlags(namedFs.FlagSet("global"))
	globalflag.AddGlobalFlags(namedFs.FlagSet("global"), cmd.Name())
	for _, f := range namedFs.FlagSets {
		fs.AddFlagSet(f)
	}

	usageFmt := "Usage:\n  %s\n"
	cols, _, _ := term.TerminalSize(cmd.OutOrStdout())
	cmd.SetUsageFunc(func(cmd *cobra.Command) error {
		fmt.Fprintf(cmd.OutOrStderr(), usageFmt, cmd.UseLine())
		cliflag.PrintSections(cmd.OutOrStderr(), namedFs, cols)
		return nil
	})
	cmd.SetHelpFunc(func(cmd *cobra.Command, args []string) {
		fmt.Fprintf(cmd.OutOrStdout(), "%s\n\n"+usageFmt, cmd.Long, cmd.UseLine())
		cliflag.PrintSections(cmd.OutOrStdout(), namedFs, cols)
	})

	return cmd
}
