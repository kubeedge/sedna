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

package runtime

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"

	sednav1 "github.com/kubeedge/sedna/pkg/apis/sedna/v1alpha1"
)

const (
	// resourceUpdateTries defines times of trying to update resource
	resourceUpdateTries = 3
)

func ResourceAwarePodScheduling(kubeClient kubernetes.Interface, deployment *appsv1.Deployment) (string, error) {
	n, err := kubeClient.CoreV1().Nodes().List(context.Background(), metav1.ListOptions{
		LabelSelector: "sedna-role=" + deployment.Spec.Template.Spec.NodeSelector["sedna-role"],
	})

	if err != nil {
		return "", err
	}

	// To load pods evenly on the retrieved nodes, we need to check their load (in terms of running pods)
	running_pods := math.MaxInt32
	var best_candidate v1.Node

	if len(n.Items) == 1 {
		best_candidate = n.Items[0]
	} else {
		for _, elem := range n.Items {
			pods, err := kubeClient.CoreV1().Pods("").List(context.Background(), metav1.ListOptions{
				FieldSelector: "spec.nodeName=" + elem.Name,
			})

			if err != nil {
				return "", err
			}

			if len(pods.Items) < running_pods {
				best_candidate = elem
				running_pods = len(pods.Items)
			}
		}
	}

	klog.Info(best_candidate)

	return "", nil
}

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

// This function returns the "sector" in which the node is deployed (edge or cloud)
func IdentifyNodeSector(kubeClient kubernetes.Interface, name string) (string, error) {
	n, err := kubeClient.CoreV1().Nodes().Get(context.Background(), name, metav1.GetOptions{})

	if err != nil {
		return "", err
	}

	// The label sedna-role=edge is assigned by default to all edge node in the Sedna cluster
	// The master node (or the cloud) usually doesn't have this flag set, so we just return "cloud"
	v, found := n.GetLabels()["sedna-role"]
	if found {
		return v, err
	} else {
		return "cloud", err
	}

}

func FindColocatedFluentdPod(kubeClient kubernetes.Interface, name string) (string, error) {
	pods, err := kubeClient.CoreV1().Pods("").List(context.Background(), metav1.ListOptions{
		FieldSelector: "spec.nodeName=" + name,
		LabelSelector: "app=" + "fluentd",
	})

	if len(pods.Items) > 0 {
		klog.Info("Found local Fluentd service with IP: " + pods.Items[0].Status.PodIP)
		return pods.Items[0].Status.PodIP, nil
	}

	return "", err
}

func FindDaemonSetByLabel(kubeClient kubernetes.Interface, namespace string, label string) ([]string, error) {
	labelSelector := fmt.Sprintf(label)
	s, err := kubeClient.AppsV1().DaemonSets(namespace).List(context.Background(), metav1.ListOptions{LabelSelector: labelSelector})

	if len(s.Items) == 0 {
		return nil, fmt.Errorf("Unable to find any DaemonSet with label %s in namespace %s", label, namespace)
	} else {
		for _, ds := range s.Items {
			ds.Spec.Template.Spec.Overhead.Pods()
		}
	}

	return nil, err
}

func FindAvailableKafkaServices(kubeClient kubernetes.Interface, name string, sector string) ([]string, []string, error) {
	s, err := kubeClient.CoreV1().Services("default").List(context.Background(), metav1.ListOptions{})
	if err != nil {
		return nil, nil, err
	}

	kafkaAddresses := []string{}
	kafkaPorts := []string{}

	// For this to work, the kafka-service has to contain an annotation field
	// containing the advertised_IP of the Kafka broker (the same as in the deployment).
	// Additionally, another label field called `sector` has to be set in the Kafka service
	// to indicate where the broker is deployed Kafka broker is at the edge.
	// Ideally, it should match the KAFKA_ADVERTISED_HOST_NAME in the deployment configuration.
	// Using the service name doesn't work using the defualt Kafka YAML file.

	if sector == "edge" {
		for _, svc := range s.Items {
			if strings.Contains(svc.GetName(), name) && svc.GetLabels()["sector"] == sector {
				klog.Info("Found Apache Kafka edge service: " + svc.GetName() + "" + svc.GetClusterName())
				kafkaAddresses = append(kafkaAddresses, svc.GetName()+"."+svc.GetNamespace())
				kafkaPorts = append(kafkaPorts, fmt.Sprint(svc.Spec.Ports[0].Port))
			}
		}
	} else {
		for _, svc := range s.Items {
			val, found := svc.GetLabels()["sector"] // If we don't find it, it's a cloud service
			if strings.Contains(svc.GetName(), name) && (!found || val == sector) {
				klog.Info("Found Apache Kafka cloud service: " + svc.GetName() + "" + svc.GetClusterName())
				kafkaAddresses = append(kafkaAddresses, svc.Annotations["advertised_ip"])
				kafkaPorts = append(kafkaPorts, fmt.Sprint(svc.Spec.Ports[0].NodePort))
			}
		}
	}

	if len(kafkaAddresses) > 0 && len(kafkaPorts) > 0 {
		return kafkaAddresses, kafkaPorts, err
	}

	return nil, nil, fmt.Errorf("can't find node ip for node %s", name)
}

// getBackoff calc the next wait time for the key
func GetBackoff(queue workqueue.RateLimitingInterface, key interface{}) time.Duration {
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

func CalcActivePodCount(pods []*v1.Pod) int32 {
	var result int32 = 0
	for _, p := range pods {
		if v1.PodSucceeded != p.Status.Phase &&
			v1.PodFailed != p.Status.Phase &&
			p.DeletionTimestamp == nil {
			result++
		}
	}
	klog.Infof("Active pods: %d", result)
	return result
}

func CalcActiveDeploymentCount(deployments []*appsv1.Deployment) int32 {
	var result int32 = 0
	var latestConditionType appsv1.DeploymentConditionType
	for _, d := range deployments {
		dConditions := d.Status.Conditions
		if len(dConditions) > 0 {
			latestConditionType = (dConditions)[len(dConditions)-1].Type
		}
		if appsv1.DeploymentProgressing == latestConditionType &&
			d.DeletionTimestamp == nil {
			result++
		}
	}
	return result
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

// ConvertMapToMetrics converts the metric map to list of resource Metric
func ConvertMapToMetrics(metric map[string]interface{}) []sednav1.Metric {
	var l []sednav1.Metric
	for k, v := range metric {
		var displayValue string
		switch t := v.(type) {
		case string:
			displayValue = t
		default:
			// ignore the json marshal error
			b, _ := json.Marshal(v)
			displayValue = string(b)
		}

		l = append(l, sednav1.Metric{Key: k, Value: displayValue})
	}
	return l
}

// RetryUpdateStatus simply retries to call the status update func
func RetryUpdateStatus(name, namespace string, updateStatusFunc func() error) error {
	var err error
	for try := 1; try <= resourceUpdateTries; try++ {
		err = updateStatusFunc()
		if err == nil {
			return nil
		}
		klog.Warningf("Error to update %s/%s status, tried %d times: %+v", namespace, name, try, err)
	}
	return err
}
