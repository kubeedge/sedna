package main

import (
	"os"

	"k8s.io/component-base/logs"

	"github.com/edgeai-neptune/neptune/cmd/neptune-lc/app"
)

func main() {
	command := app.NewLocalControllerCommand()
	logs.InitLogs()
	defer logs.FlushLogs()

	if err := command.Execute(); err != nil {
		os.Exit(1)
	}
}
