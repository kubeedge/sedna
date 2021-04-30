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

package globalmanager

import (
	"context"
	"fmt"
	"math"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
)

const (
	// DefaultBackOff is the default backoff period
	DefaultBackOff = 10 * time.Second
	// MaxBackOff is the max backoff period
	MaxBackOff          = 360 * time.Second
	statusUpdateRetries = 3

	bigModelPort int32 = 5000
)

// GetNodeIPByName get node ip by node name
func GetNodeIPByName(kubeClient kubernetes.Interface, name string) (string, error) {
	n, err := kubeClient.CoreV1().Nodes().Get(context.Background(), name, metav1.GetOptions{})
	if err != nil {
		return "", err
	}

	typeToAddress := make(map[v1.NodeAddressType]string)
	for _, addr := range n.Status.Addresses {
		typeToAddress[addr.Type] = addr.Address
	}
	address, found := typeToAddress[v1.NodeExternalIP]
	if found {
		return address, nil
	}

	address, found = typeToAddress[v1.NodeInternalIP]
	if found {
		return address, nil
	}
	return "", fmt.Errorf("can't found node ip for node %s", name)
}

// CreateKubernetesService creates a k8s service for an object given ip and port
func CreateKubernetesService(kubeClient kubernetes.Interface, object CommonInterface, inputPort int32, inputIP string) (int32, error) {
	ctx := context.Background()
	name := object.GetName()
	namespace := object.GetNamespace()
	kind := object.GroupVersionKind().Kind
	targePort := intstr.IntOrString{
		IntVal: inputPort,
	}
	serviceSpec := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Namespace:    object.GetNamespace(),
			GenerateName: name + "-" + "service" + "-",
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(object, object.GroupVersionKind()),
			},
			Labels: GenerateLabels(object),
		},
		Spec: v1.ServiceSpec{
			Selector: GenerateLabels(object),
			ExternalIPs: []string{
				inputIP,
			},
			Type: v1.ServiceTypeNodePort,
			Ports: []v1.ServicePort{
				{
					Port:       inputPort,
					TargetPort: targePort,
				},
			},
		},
	}
	service, err := kubeClient.CoreV1().Services(namespace).Create(ctx, serviceSpec, metav1.CreateOptions{})
	if err != nil {
		klog.Warningf("failed to create service for %v %v/%v, err:%s", kind, namespace, name, err)
		return 0, err
	}

	klog.V(2).Infof("Service %s is created successfully for %v %v/%v", service.Name, kind, namespace, name)
	return service.Spec.Ports[0].NodePort, nil
}

// getBackoff calc the next wait time for the key
func getBackoff(queue workqueue.RateLimitingInterface, key interface{}) time.Duration {
	exp := queue.NumRequeues(key)

	if exp <= 0 {
		return time.Duration(0)
	}

	// The backoff is capped such that 'calculated' value never overflows.
	backoff := float64(DefaultBackOff.Nanoseconds()) * math.Pow(2, float64(exp-1))
	if backoff > math.MaxInt64 {
		return MaxBackOff
	}

	calculated := time.Duration(backoff)
	if calculated > MaxBackOff {
		return MaxBackOff
	}
	return calculated
}

func calcActivePodCount(pods []*v1.Pod) int32 {
	var result int32 = 0
	for _, p := range pods {
		if v1.PodSucceeded != p.Status.Phase &&
			v1.PodFailed != p.Status.Phase &&
			p.DeletionTimestamp == nil {
			result++
		}
	}
	return result
}

// GenerateLabels generates labels for an object
func GenerateLabels(object CommonInterface) map[string]string {
	kind := object.GroupVersionKind().Kind
	group := object.GroupVersionKind().Group

	keyPrefix := strings.ToLower(kind + "." + group + "/")

	labels := make(map[string]string)
	labels[keyPrefix+"name"] = object.GetName()
	labels[keyPrefix+"uid"] = string(object.GetUID())
	return labels
}

// GenerateSelector generates the selector for an object
func GenerateSelector(object CommonInterface) (labels.Selector, error) {
	ls := &metav1.LabelSelector{
		MatchLabels: GenerateLabels(object),
	}
	return metav1.LabelSelectorAsSelector(ls)
}

// ConvertK8SValidName converts to the k8s valid name
func ConvertK8SValidName(name string) string {
	// the name(e.g. pod/volume name) should be a lowercase RFC 1123 label:
	// [a-z0-9]([-a-z0-9]*[a-z0-9])?
	// and no more than 63 characters
	limitCount := 63
	var fixName []byte
	for _, c := range []byte(strings.ToLower(name)) {
		if ('a' <= c && c <= 'z') ||
			('0' <= c && c <= '9') ||
			c == '-' {
			fixName = append(fixName, c)
			continue
		}

		// the first char not '-'
		// and no two consecutive '-'
		if len(fixName) > 0 && fixName[len(fixName)-1] != '-' {
			fixName = append(fixName, '-')
		}
	}

	// fix limitCount
	if len(fixName) > limitCount {
		fixName = fixName[:limitCount]
	}

	// fix the end character
	if len(fixName) > 0 && fixName[len(fixName)-1] == '-' {
		fixName[len(fixName)-1] = 'z'
	}

	return string(fixName)
}
