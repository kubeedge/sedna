package globalmanager

import (
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"

	neptunev1 "github.com/edgeai-neptune/neptune/pkg/apis/neptune/v1alpha1"
	clientset "github.com/edgeai-neptune/neptune/pkg/client/clientset/versioned/typed/neptune/v1alpha1"
	"github.com/edgeai-neptune/neptune/pkg/globalmanager/config"
	"github.com/edgeai-neptune/neptune/pkg/globalmanager/messagelayer"
	"github.com/edgeai-neptune/neptune/pkg/globalmanager/utils"
)

// DownstreamController watch kubernetes api server and send the controller resource change to edge
type DownstreamController struct {
	// events from watch kubernetes api server
	events chan watch.Event

	cfg *config.ControllerConfig

	client       *clientset.NeptuneV1alpha1Client
	messageLayer messagelayer.MessageLayer
}

// syncDataset syncs the dataset resources
func (dc *DownstreamController) syncDataset(eventType watch.EventType, dataset *neptunev1.Dataset) error {
	// Here only propagate to the nodes with non empty name
	nodeName := dataset.Spec.NodeName
	if len(nodeName) == 0 {
		return fmt.Errorf("empty node name")
	}

	return dc.messageLayer.SendResourceObject(nodeName, eventType, dataset)
}

// syncJointInferenceService syncs the joint-inference-service resources
func (dc *DownstreamController) syncJointInferenceService(eventType watch.EventType, joint *neptunev1.JointInferenceService) error {
	// Here only propagate to the nodes with non empty name
	nodeName := joint.Spec.EdgeWorker.NodeName
	if len(nodeName) == 0 {
		return fmt.Errorf("empty node name")
	}

	return dc.messageLayer.SendResourceObject(nodeName, eventType, joint)
}

// sync defines the entrypoint of syncing all resources
func (dc *DownstreamController) sync(stopCh <-chan struct{}) {
	for {
		select {
		case <-stopCh:
			klog.Info("Stop controller downstream loop")
			return

		case e := <-dc.events:

			var err error
			var kind, namespace, name string
			switch t := e.Object.(type) {
			case (*neptunev1.Dataset):
				// Since t.Kind may be empty,
				// we need to fix the kind here if missing.
				// more details at https://github.com/kubernetes/kubernetes/issues/3030
				if len(t.Kind) == 0 {
					t.Kind = "Dataset"
				}
				kind = t.Kind
				namespace = t.Namespace
				name = t.Name
				err = dc.syncDataset(e.Type, t)

			case (*neptunev1.JointInferenceService):
				// TODO: find a good way to avoid these duplicate codes
				if len(t.Kind) == 0 {
					t.Kind = "JointInferenceService"
				}
				kind = t.Kind
				namespace = t.Namespace
				name = t.Name
				err = dc.syncJointInferenceService(e.Type, t)

			default:
				klog.Warningf("object type: %T unsupported", e)
				continue
			}

			if err != nil {
				klog.Warningf("Error to sync %s(%s/%s), err: %+v", kind, namespace, name, err)
			} else {
				klog.V(2).Infof("synced %s(%s/%s)", kind, namespace, name)
			}
		}
	}
}

// watch function watches the crd resources which should by synced to nodes
func (dc *DownstreamController) watch(stopCh <-chan struct{}) {
	rh := cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj interface{}) {
			eventObj := obj.(runtime.Object)
			dc.events <- watch.Event{Type: watch.Added, Object: eventObj}
		},
		UpdateFunc: func(old, cur interface{}) {
			// Since we don't support the spec update operation currently,
			// so only status updates arrive here and NO propagation to edge.

			// Update:
			// We sync it to edge when using self-built websocket, and
			// this sync isn't needed when we switch out self-built websocket.
			dc.events <- watch.Event{Type: watch.Added, Object: cur.(runtime.Object)}
		},
		DeleteFunc: func(obj interface{}) {
			eventObj := obj.(runtime.Object)
			dc.events <- watch.Event{Type: watch.Deleted, Object: eventObj}
		},
	}

	client := dc.client.RESTClient()

	// make this option configurable
	resyncPeriod := time.Second * 60
	namespace := dc.cfg.Namespace

	for resourceName, object := range map[string]runtime.Object{
		"datasets":               &neptunev1.Dataset{},
		"jointinferenceservices": &neptunev1.JointInferenceService{},
	} {
		lw := cache.NewListWatchFromClient(client, resourceName, namespace, fields.Everything())
		si := cache.NewSharedInformer(lw, object, resyncPeriod)
		si.AddEventHandler(rh)
		go si.Run(stopCh)
	}
}

// Start starts the controller
func (dc *DownstreamController) Start() error {
	stopCh := dc.messageLayer.Done()

	// watch is an asynchronous call
	dc.watch(stopCh)

	// sync is a synchronous call
	go dc.sync(stopCh)

	return nil
}

func (dc *DownstreamController) GetName() string {
	return "DownstreamController"
}

// NewDownstreamController creates a controller DownstreamController from config
func NewDownstreamController(cfg *config.ControllerConfig) (FeatureControllerI, error) {
	// TODO: make bufferSize configurable
	bufferSize := 10
	events := make(chan watch.Event, bufferSize)

	crdclient, err := utils.NewCRDClient()
	if err != nil {
		return nil, fmt.Errorf("create crd client failed with error: %w", err)
	}

	dc := &DownstreamController{
		cfg:          cfg,
		events:       events,
		client:       crdclient,
		messageLayer: messagelayer.NewContextMessageLayer(),
	}

	return dc, nil
}
