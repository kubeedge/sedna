package ws

import (
	gocontext "context"
	"fmt"
	"strings"
	"sync"

	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"

	"github.com/edgeai-neptune/neptune/pkg/globalmanager/messagelayer/model"
)

type nodeMessage struct {
	nodeName string
	msg      model.Message
}

// ChannelContext is object for Context channel
type ChannelContext struct {
	ctx    gocontext.Context
	cancel gocontext.CancelFunc

	upstreamChannel chan nodeMessage
	//downstreamChannel chan nodeMessage

	// downstream map
	// nodeName => queue
	nodeQueue sync.Map
	nodeStore sync.Map
}

var (
	// singleton
	context *ChannelContext
)

func init() {
	context = NewChannelContext()
}

func NewChannelContext() *ChannelContext {
	upstreamSize := 1000
	upstreamChannel := make(chan nodeMessage, upstreamSize)
	//downstreamChannel := make(chan nodeMessage, bufferSize)

	ctx, cancel := gocontext.WithCancel(gocontext.Background())
	return &ChannelContext{
		upstreamChannel: upstreamChannel,
		//		downstreamChannel: downstreamChannel,
		// nodeMessageChannelMap: make(map[string]chan model.Message),
		ctx:    ctx,
		cancel: cancel,
	}
}

func getMsgKey(obj interface{}) (string, error) {
	msg := obj.(*model.Message)

	kind := msg.ResourceKind
	namespace := msg.Namespace
	name := msg.ResourceName
	return strings.Join([]string{kind, namespace, name}, "/"), nil
}

func GetNodeQueue(nodeName string) workqueue.RateLimitingInterface {
	q, ok := context.nodeQueue.Load(nodeName)
	if !ok {
		newQ := workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), nodeName)
		q, _ = context.nodeQueue.LoadOrStore(nodeName, newQ)
	}
	return q.(workqueue.RateLimitingInterface)
}

func GetNodeStore(nodeName string) cache.Store {
	s, ok := context.nodeStore.Load(nodeName)
	if !ok {
		newS := cache.NewStore(getMsgKey)
		s, _ = context.nodeStore.LoadOrStore(nodeName, newS)
	}
	return s.(cache.Store)
}

func SendToEdge(nodeName string, msg *model.Message) error {
	q := GetNodeQueue(nodeName)
	key, _ := getMsgKey(msg)
	q.Add(key)

	s := GetNodeStore(nodeName)
	return s.Add(msg)
}

func ReceiveFromEdge() (nodeName string, msg model.Message, err error) {
	nodeMsg := <-context.upstreamChannel
	nodeName = nodeMsg.nodeName
	msg = nodeMsg.msg
	return
}

func SendToCloud(nodeName string, msg model.Message) error {
	context.upstreamChannel <- nodeMessage{nodeName, msg}
	return nil
}

func Done() <-chan struct{} {
	return context.ctx.Done()
}

type ReadMsgFunc func() (model.Message, error)
type WriteMsgFunc func(model.Message) error

func AddNode(nodeName string, read ReadMsgFunc, write WriteMsgFunc, closeCh chan struct{}) {
	GetNodeQueue(nodeName)
	GetNodeStore(nodeName)

	go func() {
		// read loop
		var msg model.Message
		var err error
		for {
			msg, err = read()
			if err != nil {
				break
			}
			klog.V(4).Infof("received msg from %s: %+v", nodeName, msg)
			_ = SendToCloud(nodeName, msg)
		}
		closeCh <- struct{}{}
		klog.Errorf("read loop of node %s closed, due to: %+v", nodeName, err)
	}()

	go func() {
		// write loop
		q := GetNodeQueue(nodeName)
		s := GetNodeStore(nodeName)
		var err error
		for {
			key, shutdown := q.Get()
			if shutdown {
				err = fmt.Errorf("node queue for node %s shutdown", nodeName)
				break
			}
			obj, exists, _ := s.GetByKey(key.(string))
			if !exists {
				klog.Warningf("key %s not exists in node store %s", key, nodeName)
				q.Forget(key)
				q.Done(key)
				continue
			}
			msg := obj.(*model.Message)
			err = write(*msg)
			klog.V(4).Infof("writing msg to %s: %+v", nodeName, msg)
			if err != nil {
				klog.Warningf("failed to write key %s to node %s, requeue", key, nodeName)
				q.AddRateLimited(key)
				q.Forget(key)
				q.Done(key)
				break
			}
			klog.Infof("write key %s to node %s successfully", key, nodeName)
			_ = s.Delete(msg)
			q.Forget(key)
			q.Done(key)
		}
		closeCh <- struct{}{}
		klog.Errorf("write loop of node %s closed, due to: %+v", nodeName, err)
	}()
}
