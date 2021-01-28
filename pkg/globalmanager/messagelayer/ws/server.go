package ws

import (
	"net/http"

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

func (srv *Server) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	nodeName := req.Header.Get("Node-Name")
	wsConn := srv.upgrade(w, req)
	if wsConn == nil {
		klog.Errorf("failed to upgrade to websocket for node %s", nodeName)
		return
	}

	// serve connection
	nodeClient := &nodeClient{conn: wsConn, req: req}
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
	nodeName := nc.req.Header.Get("Node-Name")
	nc.nodeName = nodeName
	klog.Infof("established connection for node %s", nodeName)
	// nc.conn.SetCloseHandler
	closeCh := make(chan struct{}, 2)
	AddNode(nodeName, nc.readOneMsg, nc.writeOneMsg, closeCh)
	<-closeCh

	klog.Infof("closed connection for node %s", nodeName)
	_ = nc.conn.Close()
}
