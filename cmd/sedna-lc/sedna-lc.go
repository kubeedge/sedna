package main

import (
	"os"

	"k8s.io/component-base/logs"

	"github.com/kubeedge/sedna/cmd/sedna-lc/app"
)

func main() {
	command := app.NewLocalControllerCommand()
	logs.InitLogs()
	defer logs.FlushLogs()

	if err := command.Execute(); err != nil {
		os.Exit(1)
	}
}
