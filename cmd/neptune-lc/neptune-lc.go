package main

import (
	"os"

	"github.com/edgeai-neptune/neptune/cmd/neptune-lc/app"
)

func main() {
	command := app.NewLocalControllerCommand()
	if err := command.Execute(); err != nil {
		os.Exit(1)
	}
}
