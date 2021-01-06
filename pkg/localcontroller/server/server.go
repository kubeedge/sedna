package server

import (
	"fmt"
	"net/http"

	"github.com/emicklei/go-restful/v3"
	"k8s.io/klog/v2"

	"github.com/edgeai-neptune/neptune/cmd/neptune-lc/app/options"
	"github.com/edgeai-neptune/neptune/pkg/localcontroller/common/constants"
	"github.com/edgeai-neptune/neptune/pkg/localcontroller/manager"
)

// Server defines server
type Server struct {
	Port            string
	Resource        *Resource
	FeatureManagers featureManagerMap
}

// Resource defines resource
type Resource struct {
	Worker map[string]manager.WorkerMessage
}

// ResponseMessage defines send message to worker
type ResponseMessage struct {
	Status  int
	Message string
}

type featureManagerMap map[string]manager.FeatureManager

// NewServer create a new LC server
func NewServer(options *options.LocalControllerOptions, featureManagers ...manager.FeatureManager) *Server {
	s := Server{
		Port: options.BindPort,
	}

	fms := featureManagerMap{}
	for _, m := range featureManagers {
		if err := m.Start(); err != nil {
			klog.Errorf("start %s manager failed, error %v", m.GetKind(), err)
		}
		fms[m.GetKind()] = m
	}

	s.FeatureManagers = fms

	return &s
}

// register registers api
func (s *Server) register(container *restful.Container) {
	ws := new(restful.WebService)
	ws.Path(fmt.Sprintf("/%s", constants.ServerRootPath)).
		Consumes(restful.MIME_XML, restful.MIME_JSON).
		Produces(restful.MIME_JSON, restful.MIME_XML)

	ws.Route(ws.POST("/workers/{worker-name}/info").
		To(s.messageHandler).
		Doc("receive worker message"))
	container.Add(ws)
}

// reply replies message to the worker
func (s *Server) reply(response *restful.Response, statusCode int, msg string) error {
	err := response.WriteHeaderAndEntity(statusCode, ResponseMessage{
		Status:  statusCode,
		Message: msg,
	})

	if err != nil {
		klog.Errorf("the value could not be written on the response, error: %v", err)
		return err
	}

	return nil
}

// messageHandler handles message from the worker
func (s *Server) messageHandler(request *restful.Request, response *restful.Response) {
	var err error
	workerName := request.PathParameter("worker-name")
	workerMessage := manager.WorkerMessage{}

	err = request.ReadEntity(&workerMessage)
	if workerMessage.Name != workerName || err != nil {
		var msg string

		if workerMessage.Name != workerName {
			msg = fmt.Sprintf("worker name(name=%s) in the api is different from that(name=%s) in the message body",
				workerName, workerMessage.Name)
		} else {
			msg = fmt.Sprintf("read worker(name=%s) message body failed, error: %v", workerName, err)
		}

		klog.Errorf(msg)
		err = s.reply(response, http.StatusBadRequest, msg)
		if err != nil {
			klog.Errorf("reply messge to worker(name=%s) failed, error: %v", workerName, err)
		}

		return
	}

	if m, ok := s.FeatureManagers[workerMessage.OwnerKind]; ok {
		m.AddWorkerMessageToChannel(workerMessage)
	}

	err = s.reply(response, http.StatusOK, "OK")
	if err != nil {
		klog.Errorf("reply message to worker(name=%s) failed, error: %v", workerName, err)
		return
	}
}

// Start starts server
func (s *Server) Start() {
	wsContainer := restful.NewContainer()
	resource := Resource{map[string]manager.WorkerMessage{}}
	s.Resource = &resource
	s.register(wsContainer)

	server := &http.Server{Addr: fmt.Sprintf(":%s", s.Port), Handler: wsContainer}

	klog.Infof("server binds port %s successfully", s.Port)
	klog.Fatal(server.ListenAndServe())
}
