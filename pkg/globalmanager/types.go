package globalmanager

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// ContainerPara describes initial values need by creating a pod
type ContainerPara struct {
	volumeMountList []string
	volumeList      []string
	volumeMapName   []string
	env             map[string]string
	frameName       string
	frameVersion    string
	scriptBootFile  string
	nodeName        string
}

// CommonInterface describes the commom interface of CRs
type CommonInterface interface {
	metav1.Object
	schema.ObjectKind
}

// FeatureControllerI defines the interface of an AI Feature controller
type FeatureControllerI interface {
	Start() error
	GetName() string
}

type Model struct {
	Format  string                 `json:"format,omitempty"`
	URL     string                 `json:"url,omitempty"`
	Metrics map[string]interface{} `json:"metrics,omitempty"`
}
