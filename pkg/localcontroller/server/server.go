/*
Copyright 2021 The KubeEdge Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package server

import (
	"fmt"
	"net/http"

	"github.com/emicklei/go-restful/v3"
	"k8s.io/klog/v2"

	"github.com/kubeedge/sedna/cmd/sedna-lc/app/options"
	"github.com/kubeedge/sedna/pkg/localcontroller/common/constants"
	"github.com/kubeedge/sedna/pkg/localcontroller/managers"
	workertypes "github.com/kubeedge/sedna/pkg/localcontroller/worker"
)

// Server defines server
type Server struct {
	Port     string
	Resource *Resource
	fmm      featureManagerMap
}

// Resource defines resource
type Resource struct {
	Worker map[string]workertypes.MessageContent
}

// ResponseMessage defines send message to worker
type ResponseMessage struct {
	Status  int
	Message string
}

type featureManagerMap map[string]managers.FeatureManager

// New creates a new LC server
func New(options *options.LocalControllerOptions) *Server {
	s := Server{
		Port: options.BindPort,
	}

	s.fmm = featureManagerMap{}

	return &s
}

func (s *Server) AddFeatureManager(m managers.FeatureManager) {
	s.fmm[m.GetName()] = m
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
	workerMessage := workertypes.MessageContent{}

	err = request.ReadEntity(&workerMessage)
	if workerMessage.Name != workerName || err != nil {
		var msg string
		if workerMessage.Name != workerName {
			msg = fmt.Sprintf("worker name(name=%s) in the api is different from that(name=%s) in the message body",
				workerName, workerMessage.Name)
		} else {
			msg = fmt.Sprintf("read worker(name=%s) message body failed, error: %v", workerName, err)
		}

		klog.Error(msg)
		err = s.reply(response, http.StatusBadRequest, msg)
		if err != nil {
			klog.Errorf("reply messge to worker(name=%s) failed, error: %v", workerName, err)
		}

		return
	}

	if m, ok := s.fmm[workerMessage.OwnerKind]; ok {
		m.AddWorkerMessage(workerMessage)
	}

	err = s.reply(response, http.StatusOK, "OK")
	if err != nil {
		klog.Errorf("reply message to worker(name=%s) failed, error: %v", workerName, err)
		return
	}
}

// ListenAndServe starts server
func (s *Server) ListenAndServe() {
	wsContainer := restful.NewContainer()
	resource := Resource{map[string]workertypes.MessageContent{}}
	s.Resource = &resource
	s.register(wsContainer)

	server := &http.Server{Addr: fmt.Sprintf(":%s", s.Port), Handler: wsContainer}

	klog.Infof("server binds port %s successfully", s.Port)
	klog.Fatal(server.ListenAndServe())
}
