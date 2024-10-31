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

package ws

import (
	"fmt"
	"net/http"
	"strings"

	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/klog/v2"

	"github.com/gorilla/websocket"

	"github.com/kubeedge/sedna/pkg/globalmanager/messagelayer/model"
)

// Server defines websocket protocol server
type Server struct {
	server *http.Server
}

// NewServer creates a websocket server
func NewServer(address string) *Server {
	server := http.Server{
		Addr: address,
	}

	wsServer := &Server{
		server: &server,
	}
	http.HandleFunc("/", wsServer.ServeHTTP)
	return wsServer
}

func (srv *Server) upgrade(w http.ResponseWriter, r *http.Request) *websocket.Conn {
	upgrader := websocket.Upgrader{
		ReadBufferSize:  1024,
		WriteBufferSize: 1024,
	}
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		return nil
	}
	return conn
}

func validateNodeName(rawNodeName string) (nodeName string, err error) {
	// Node name follows DNS Subdomain constraint
	// https://kubernetes.io/docs/concepts/overview/working-with-objects/names/#dns-subdomain-names
	errs := validation.IsDNS1123Subdomain(rawNodeName)
	nodeName = strings.ReplaceAll(rawNodeName, "\n", "")
	nodeName = strings.ReplaceAll(nodeName, "\r", "")
	if len(errs) > 0 {
		err = fmt.Errorf("invalid node name %s: %s", nodeName, strings.Join(errs, ","))
		nodeName = ""
	}
	return
}

func (srv *Server) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	rawNodeName := req.Header.Get("Node-Name")

	nodeName, err := validateNodeName(rawNodeName)
	if err != nil {
		klog.Warningf("closing the connection, due to: %v", err)
		return
	}
	wsConn := srv.upgrade(w, req)
	if wsConn == nil {
		klog.Errorf("failed to upgrade to websocket for node %s", nodeName)
		return
	}

	// serve connection
	nodeClient := &nodeClient{conn: wsConn, req: req, nodeName: nodeName}
	go nodeClient.Serve()
}

// ListenAndServe listens and serves the server
func (srv *Server) ListenAndServe() error {
	return srv.server.ListenAndServe()
}

// Close closes the server
func (srv *Server) Close() error {
	if srv.server != nil {
		return srv.server.Close()
	}
	return nil
}

type nodeClient struct {
	conn     *websocket.Conn
	req      *http.Request
	nodeName string
}

func (nc *nodeClient) readOneMsg() (model.Message, error) {
	var msg model.Message

	err := nc.conn.ReadJSON(&msg)
	if err != nil {
		return msg, err
	}

	return msg, nil
}

func (nc *nodeClient) writeOneMsg(msg model.Message) error {
	return nc.conn.WriteJSON(&msg)
}

func (nc *nodeClient) Serve() {
	nodeName := nc.nodeName
	klog.Infof("established connection for node %s", nodeName)
	closeCh := make(chan struct{}, 2)
	AddNode(nodeName, nc.readOneMsg, nc.writeOneMsg, closeCh)
	<-closeCh

	klog.Infof("closed connection for node %s", nodeName)
	_ = nc.conn.Close()
}
