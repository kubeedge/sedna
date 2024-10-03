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

// GetBackoff calc the next wait time for the key
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
	var result int32
	for _, p := range pods {
		if v1.PodSucceeded != p.Status.Phase &&
			v1.PodFailed != p.Status.Phase &&
			p.DeletionTimestamp == nil {
			result++
		}
	}
	return result
}

func CalcActiveDeploymentCount(deployments []*appsv1.Deployment) int32 {
	var result int32
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
