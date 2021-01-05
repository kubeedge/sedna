package wsclient

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"time"

	"github.com/gorilla/websocket"
	"k8s.io/klog/v2"

	"github.com/edgeai-neptune/neptune/cmd/neptune-lc/app/options"
	"github.com/edgeai-neptune/neptune/pkg/localcontroller/common/constants"
)

// MessageHandler defines message handler function
type MessageHandler func(*Message)

// Client defines a client
type Client struct {
	Options             *options.LocalControllerOptions
	WSConnection        *WSConnection
	SubscribeMessageMap map[string]MessageHandler
	SendMessageChannel  chan Message
	ReconnectChannel    chan struct{}
}

// Message defines message
type Message struct {
	Header  MessageHeader `json:"header"`
	Content []byte        `json:"content"`
}

// MessageHeader define header of message
type MessageHeader struct {
	Namespace    string `json:"namespace"`
	ResourceKind string `json:"resourceKind"`
	ResourceName string `json:"resourceName"`
	Operation    string `json:"operation"`
}

// WSConnection defines conn
type WSConnection struct {
	WSConn *websocket.Conn
}

const (
	// RetryCount is count of retrying to connecting to global manager
	RetryCount = 5
	// RetryConnectIntervalSeconds is interval time of retrying to connecting to global manager
	RetryConnectIntervalSeconds = 5
	// MessageChannelCacheSize is size of channel cache
	MessageChannelCacheSize = 100
)

// NewClient creates client
func NewClient(options *options.LocalControllerOptions) *Client {
	c := Client{
		Options:             options,
		SubscribeMessageMap: make(map[string]MessageHandler),
		SendMessageChannel:  make(chan Message, MessageChannelCacheSize),
	}

	return &c
}

// Subscribe registers in client
func (c *Client) Subscribe(resource string, handler MessageHandler) error {
	if c.SubscribeMessageMap[resource] == nil {
		c.SubscribeMessageMap[resource] = handler
	} else {
		klog.Warningf("%s had been registered in websocket client", resource)
	}

	return nil
}

// handleReceivedMessage handles received message
func (c *Client) handleReceivedMessage(stop chan struct{}) {
	defer func() {
		stop <- struct{}{}
	}()

	ws := c.WSConnection.WSConn

	for {
		message := Message{}
		if err := ws.ReadJSON(&message); err != nil {
			klog.Errorf("client received message from global manager(address: %s) failed, error: %v",
				c.Options.GMAddr, err)
			return
		}

		klog.V(2).Infof("client received message header: %+v from global manager(address: %s)",
			message.Header, c.Options.GMAddr)
		klog.V(4).Infof("client received message content: %s from global manager(address: %s)",
			message.Content, c.Options.GMAddr)

		handler := c.SubscribeMessageMap[message.Header.ResourceKind]
		if handler != nil {
			go handler(&message)
		} else {
			klog.Errorf("%s hadn't registered in websocket client", message.Header.ResourceKind)
		}
	}
}

// WriteMessage saves message in a queue
func (c *Client) WriteMessage(messageBody interface{}, messageHeader MessageHeader) error {
	content, err := json.Marshal(&messageBody)
	if err != nil {
		return err
	}

	message := Message{
		Content: content,
		Header:  messageHeader,
	}

	c.SendMessageChannel <- message

	return nil
}

// sendMessage sends the message through the connection
func (c *Client) sendMessage(stop chan struct{}) {
	defer func() {
		stop <- struct{}{}
	}()

	messageChannel := c.SendMessageChannel
	ws := c.WSConnection.WSConn

	for {
		message, ok := <-messageChannel
		if !ok {
			return
		}

		if err := ws.WriteJSON(&message); err != nil {
			klog.Errorf("client sent message to global manager(address: %s) failed, error: %v",
				c.Options.GMAddr, err)

			c.SendMessageChannel <- message

			return
		}

		klog.V(2).Infof("client sent message header: %+v to global manager(address: %s)",
			message.Header, c.Options.GMAddr)
		klog.V(4).Infof("client sent message content: %s to global manager(address: %s)",
			message.Content, c.Options.GMAddr)
	}
}

// connect tries to connect remote server
func (c *Client) connect() error {
	header := http.Header{}
	header.Add(constants.WSHeaderNodeName, c.Options.NodeName)
	u := url.URL{Scheme: constants.WSScheme, Host: c.Options.GMAddr, Path: "/"}

	klog.Infof("client starts to connect global manager(address: %s)", c.Options.GMAddr)

	for i := 0; i < RetryCount; i++ {
		wsConn, _, err := websocket.DefaultDialer.Dial(u.String(), header)

		if err == nil {
			if errW := wsConn.WriteJSON(&MessageHeader{}); errW != nil {
				return errW
			}

			c.WSConnection = &WSConnection{WSConn: wsConn}
			klog.Infof("websocket connects global manager(address: %s) successful", c.Options.GMAddr)

			return nil
		}

		klog.Errorf("client tries to connect global manager(address: %s) failed, error: %v",
			c.Options.GMAddr, err)

		time.Sleep(time.Duration(RetryConnectIntervalSeconds) * time.Second)
	}

	errorMsg := fmt.Errorf("max retry count reached when connecting global manager(address: %s)",
		c.Options.GMAddr)
	klog.Errorf("%v", errorMsg)

	return errorMsg
}

// Start starts websocket client
func (c *Client) Start() error {
	go c.reconnect()

	return nil
}

// reconnect reconnects global manager
func (c *Client) reconnect() {
	for {
		if err := c.connect(); err != nil {
			continue
		}
		ws := c.WSConnection.WSConn

		stop := make(chan struct{}, 2)
		go c.handleReceivedMessage(stop)
		go c.sendMessage(stop)
		<-stop

		_ = ws.Close()
	}
}
