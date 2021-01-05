package globalmanager

import (
	"fmt"
	"os"

	"k8s.io/klog/v2"

	"github.com/edgeai-neptune/neptune/pkg/globalmanager/config"
	websocket "github.com/edgeai-neptune/neptune/pkg/globalmanager/messagelayer/ws"
)

// NeptuneController
type NeptuneController struct {
	Config *config.ControllerConfig
}

func NewController(cc *config.ControllerConfig) *NeptuneController {
	config.InitConfigure(cc)
	return &NeptuneController{
		Config: cc,
	}
}

// Start controller
func (c *NeptuneController) Start() {
	type newFunc func(cfg *config.ControllerConfig) (FeatureControllerI, error)

	for _, featureFunc := range []newFunc{
		NewUpstreamController,
		NewDownstreamController,
		NewJointController,
	} {
		f, _ := featureFunc(c.Config)
		err := f.Start()
		if err != nil {
			klog.Warningf("failed to start controller %s: %+v", f.GetName(), err)
		} else {
			klog.Infof("started controller %s", f.GetName())
		}
	}

	addr := fmt.Sprintf("%s:%d", c.Config.WebSocket.Address, c.Config.WebSocket.Port)

	ws := websocket.NewServer(addr)
	err := ws.ListenAndServe()
	if err != nil {
		klog.Fatalf("failed to listen websocket at %s", addr)
		os.Exit(1)
	}
}
