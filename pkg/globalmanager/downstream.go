package globalmanager

import (
	"context"
	"fmt"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"

	sednav1 "github.com/kubeedge/sedna/pkg/apis/sedna/v1alpha1"
	clientset "github.com/kubeedge/sedna/pkg/client/clientset/versioned/typed/sedna/v1alpha1"
	"github.com/kubeedge/sedna/pkg/globalmanager/config"
	"github.com/kubeedge/sedna/pkg/globalmanager/messagelayer"
	"github.com/kubeedge/sedna/pkg/globalmanager/utils"
)

// DownstreamController watch kubernetes api server and send the controller resource change to edge
type DownstreamController struct {
	// events from watch kubernetes api server
	events chan watch.Event

	cfg *config.ControllerConfig

	client       *clientset.SednaV1alpha1Client
	messageLayer messagelayer.MessageLayer
}

// syncDataset syncs the dataset resources
func (dc *DownstreamController) syncDataset(eventType watch.EventType, dataset *sednav1.Dataset) error {
	// Here only propagate to the nodes with non empty name
	nodeName := dataset.Spec.NodeName
	if len(nodeName) == 0 {
		return fmt.Errorf("empty node name")
	}

	return dc.messageLayer.SendResourceObject(nodeName, eventType, dataset)
}

// syncJointInferenceService syncs the joint-inference-service resources
func (dc *DownstreamController) syncJointInferenceService(eventType watch.EventType, joint *sednav1.JointInferenceService) error {
	// Here only propagate to the nodes with non empty name
	nodeName := joint.Spec.EdgeWorker.NodeName
	if len(nodeName) == 0 {
		return fmt.Errorf("empty node name")
	}

	return dc.messageLayer.SendResourceObject(nodeName, eventType, joint)
}

// syncFederatedLearningJob syncs the federated resources
func (dc *DownstreamController) syncFederatedLearningJob(eventType watch.EventType, job *sednav1.FederatedLearningJob) error {
	// broadcast to all nodes specified in spec
	nodeset := make(map[string]bool)
	for _, trainingWorker := range job.Spec.TrainingWorkers {
		// Here only propagate to the nodes with non empty name
		if len(trainingWorker.NodeName) > 0 {
			nodeset[trainingWorker.NodeName] = true
		}
	}

	for nodeName := range nodeset {
		dc.messageLayer.SendResourceObject(nodeName, eventType, job)
	}
	return nil
}

// syncModelWithName will sync the model to the specified node.
// Now called when creating the incrementaljob.
func (dc *DownstreamController) syncModelWithName(nodeName, modelName, namespace string) error {
	model, err := dc.client.Models(namespace).Get(context.TODO(), modelName, metav1.GetOptions{})
	if err != nil {
		// TODO: maybe use err.ErrStatus.Code == 404
		return fmt.Errorf("model(%s/%s) not found", namespace, modelName)
	}

	// Since model.Kind may be empty,
	// we need to fix the kind here if missing.
	// more details at https://github.com/kubernetes/kubernetes/issues/3030
	if len(model.Kind) == 0 {
		model.Kind = "Model"
	}

	dc.messageLayer.SendResourceObject(nodeName, watch.Added, model)
	return nil
}

// syncIncrementalJob syncs the incremental learning jobs
func (dc *DownstreamController) syncIncrementalJob(eventType watch.EventType, job *sednav1.IncrementalLearningJob) error {
	// Here only propagate to the nodes with non empty name
	nodeName := job.Spec.NodeName
	if len(nodeName) == 0 {
		return fmt.Errorf("empty node name")
	}

	// Sync the model info to edgenode when the job is created
	if eventType == watch.Added {
		models := make(map[string]bool)
		for _, modelName := range []string{
			job.Spec.InitialModel.Name,
			job.Spec.DeploySpec.Model.Name,
		} {
			models[modelName] = true
		}

		for modelName := range models {
			err := dc.syncModelWithName(nodeName, modelName, job.Namespace)
			if err != nil {
				klog.Warningf("Error to sync model %s when sync incremental learning job %s to node %s: %v", modelName, job.Name, nodeName, err)
			}
		}
	} else if eventType == watch.Deleted {
		// noop
	}

	dc.messageLayer.SendResourceObject(nodeName, eventType, job)
	return nil
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
			case (*sednav1.Dataset):
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

			case (*sednav1.JointInferenceService):
				// TODO: find a good way to avoid these duplicate codes
				if len(t.Kind) == 0 {
					t.Kind = "JointInferenceService"
				}
				kind = t.Kind
				namespace = t.Namespace
				name = t.Name
				err = dc.syncJointInferenceService(e.Type, t)

			case (*sednav1.FederatedLearningJob):
				if len(t.Kind) == 0 {
					t.Kind = "FederatedLearningJob"
				}
				kind = t.Kind
				namespace = t.Namespace
				name = t.Name
				err = dc.syncFederatedLearningJob(e.Type, t)

			case (*sednav1.IncrementalLearningJob):
				if len(t.Kind) == 0 {
					t.Kind = "IncrementalLearningJob"
				}
				kind = t.Kind
				namespace = t.Namespace
				name = t.Name
				err = dc.syncIncrementalJob(e.Type, t)

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

	// TODO: use the informer
	for resourceName, object := range map[string]runtime.Object{
		"datasets":                &sednav1.Dataset{},
		"jointinferenceservices":  &sednav1.JointInferenceService{},
		"federatedlearningjobs":   &sednav1.FederatedLearningJob{},
		"incrementallearningjobs": &sednav1.IncrementalLearningJob{},
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

// GetName returns the name of the downstream controller
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
