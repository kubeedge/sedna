package gmclient

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"time"

	"github.com/gorilla/websocket"
	"k8s.io/klog/v2"

	"github.com/kubeedge/sedna/cmd/sedna-lc/app/options"
	"github.com/kubeedge/sedna/pkg/localcontroller/common/constants"
)

// wsClient defines a websocket client
type wsClient struct {
	Options             *options.LocalControllerOptions
	WSConnection        *WSConnection
	SubscribeMessageMap map[string]MessageResourceHandler
	SendMessageChannel  chan Message
	ReconnectChannel    chan struct{}
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

// NewWebSocketClient creates client
func NewWebSocketClient(options *options.LocalControllerOptions) ClientI {
	c := wsClient{
		Options:             options,
		SubscribeMessageMap: make(map[string]MessageResourceHandler),
		SendMessageChannel:  make(chan Message, MessageChannelCacheSize),
	}

	return &c
}

// Subscribe registers in client
func (c *wsClient) Subscribe(m MessageResourceHandler) error {
	name := m.GetName()
	if c.SubscribeMessageMap[name] == nil {
		c.SubscribeMessageMap[name] = m
	} else {
		klog.Warningf("%s had been registered in websocket client", name)
	}

	return nil
}

// handleReceivedMessage handles received message
func (c *wsClient) handleReceivedMessage(stop chan struct{}) {
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

		m := c.SubscribeMessageMap[message.Header.ResourceKind]
		if m != nil {
			go func() {
				var err error
				switch message.Header.Operation {
				case InsertOperation:
					err = m.Insert(&message)

				case DeleteOperation:
					err = m.Delete(&message)
				default:
					err = fmt.Errorf("unknown operation: %s", message.Header.Operation)
				}
				if err != nil {
					klog.Errorf("failed to handle message(%+v): %v", message.Header, err)
				}
			}()
		} else {
			klog.Errorf("%s hadn't registered in websocket client", message.Header.ResourceKind)
		}
	}
}

// WriteMessage saves message in a queue
func (c *wsClient) WriteMessage(messageBody interface{}, messageHeader MessageHeader) error {
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
func (c *wsClient) sendMessage(stop chan struct{}) {
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
func (c *wsClient) connect() error {
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
func (c *wsClient) Start() error {
	go c.reconnect()

	return nil
}

// reconnect reconnects global manager
func (c *wsClient) reconnect() {
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
