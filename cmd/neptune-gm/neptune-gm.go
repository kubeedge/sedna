package main

import (
	"os"

	"k8s.io/component-base/logs"

	"github.com/edgeai-neptune/neptune/cmd/neptune-gm/app"
)

func main() {
	command := app.NewControllerCommand()
	logs.InitLogs()
	defer logs.FlushLogs()

	if err := command.Execute(); err != nil {
		os.Exit(1)
	}
}
