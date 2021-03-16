/*
Copyright 2021 The KubeEdge Authors.
Copyright 2014 The Kubernetes Authors.

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

package framework

import (
	"context"
	"fmt"
	"math/rand"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

// RunID is a unique identifier of the e2e run.
// Beware that this ID is not the same for all tests in the e2e run, because each Ginkgo node creates it separately.
var RunID = uuid.NewUUID()

// DeleteNamespaces deletes all namespaces that match the given delete and skip filters.
// Filter is by simple strings.Contains; first skip filter, then delete filter.
// Returns the list of deleted namespaces or an error.
func DeleteNamespaces(c clientset.Interface, deleteFilter, skipFilter []string) ([]string, error) {
	ginkgo.By("Deleting namespaces")
	nsList, err := c.CoreV1().Namespaces().List(context.TODO(), metav1.ListOptions{})
	ExpectNoError(err, "Failed to get namespace list")
	var deleted []string
	var wg sync.WaitGroup
OUTER:
	for _, item := range nsList.Items {
		for _, pattern := range skipFilter {
			if strings.Contains(item.Name, pattern) {
				continue OUTER
			}
		}
		if deleteFilter != nil {
			var shouldDelete bool
			for _, pattern := range deleteFilter {
				if strings.Contains(item.Name, pattern) {
					shouldDelete = true
					break
				}
			}
			if !shouldDelete {
				continue OUTER
			}
		}
		wg.Add(1)
		deleted = append(deleted, item.Name)
		go func(nsName string) {
			defer wg.Done()
			defer ginkgo.GinkgoRecover()
			gomega.Expect(c.CoreV1().Namespaces().Delete(context.TODO(), nsName, metav1.DeleteOptions{})).To(gomega.Succeed())
			Logf("namespace : %v api call to delete is complete ", nsName)
		}(item.Name)
	}
	wg.Wait()
	return deleted, nil
}

// WaitForNamespacesDeleted waits for the namespaces to be deleted.
func WaitForNamespacesDeleted(c clientset.Interface, namespaces []string, timeout time.Duration) error {
	ginkgo.By(fmt.Sprintf("Waiting for namespaces %+v to vanish", namespaces))
	nsMap := map[string]bool{}
	for _, ns := range namespaces {
		nsMap[ns] = true
	}
	//Now POLL until all namespaces have been eradicated.
	return wait.Poll(2*time.Second, timeout,
		func() (bool, error) {
			nsList, err := c.CoreV1().Namespaces().List(context.TODO(), metav1.ListOptions{})
			if err != nil {
				return false, err
			}
			for _, item := range nsList.Items {
				if _, ok := nsMap[item.Name]; ok {
					return false, nil
				}
			}
			return true, nil
		})
}

// CreateTestingNS should be used by every test, note that we append a common prefix to the provided test name.
// Please see NewFramework instead of using this directly.
func CreateTestingNS(baseName string, c clientset.Interface, labels map[string]string) (*v1.Namespace, error) {
	if labels == nil {
		labels = map[string]string{}
	}
	labels["e2e-run"] = string(RunID)

	// We don't use ObjectMeta.GenerateName feature, as in case of API call
	// failure we don't know whether the namespace was created and what is its
	// name.
	name := fmt.Sprintf("%v-%v", baseName, RandomSuffix())

	namespaceObj := &v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: "",
			Labels:    labels,
		},
		Status: v1.NamespaceStatus{},
	}
	// Be robust about making the namespace creation call.
	var got *v1.Namespace
	if err := wait.PollImmediate(2*time.Second, 30*time.Second, func() (bool, error) {
		var err error
		got, err = c.CoreV1().Namespaces().Create(context.TODO(), namespaceObj, metav1.CreateOptions{})
		if err != nil {
			if apierrors.IsAlreadyExists(err) {
				// regenerate on conflict
				Logf("Namespace name %q was already taken, generate a new name and retry", namespaceObj.Name)
				namespaceObj.Name = fmt.Sprintf("%v-%v", baseName, RandomSuffix())
			} else {
				Logf("Unexpected error while creating namespace: %v", err)
			}
			return false, nil
		}
		return true, nil
	}); err != nil {
		return nil, err
	}

	return got, nil
}

// restclientConfig returns a config holds the information needed to build connection to kubernetes clusters.
func restclientConfig(kubeContext string) (*clientcmdapi.Config, error) {
	Logf(">>> kubeConfig: %s", TestContext.KubeConfig)
	if TestContext.KubeConfig == "" {
		return nil, fmt.Errorf("KubeConfig must be specified to load client config")
	}
	c, err := clientcmd.LoadFromFile(TestContext.KubeConfig)
	if err != nil {
		return nil, fmt.Errorf("error loading KubeConfig: %v", err.Error())
	}
	if kubeContext != "" {
		Logf(">>> kubeContext: %s", kubeContext)
		c.CurrentContext = kubeContext
	}
	return c, nil
}

// LoadConfig returns a config for a rest client with the UserAgent set to include the current test name.
func LoadConfig() (config *restclient.Config, err error) {
	defer func() {
		if err == nil && config != nil {
			testDesc := ginkgo.CurrentGinkgoTestDescription()
			if len(testDesc.ComponentTexts) > 0 {
				componentTexts := strings.Join(testDesc.ComponentTexts, " ")
				config.UserAgent = fmt.Sprintf("%s -- %s", rest.DefaultKubernetesUserAgent(), componentTexts)
			}
		}
	}()

	c, err := restclientConfig("")
	if err != nil {
		if TestContext.KubeConfig == "" {
			return restclient.InClusterConfig()
		}
		return nil, err
	}

	return clientcmd.NewDefaultClientConfig(*c, &clientcmd.ConfigOverrides{ClusterInfo: clientcmdapi.Cluster{Server: TestContext.Master}}).ClientConfig()
}

// LoadClientset returns clientset for connecting to kubernetes clusters.
func LoadClientset() (*clientset.Clientset, error) {
	config, err := LoadConfig()
	if err != nil {
		return nil, fmt.Errorf("error creating client: %v", err.Error())
	}
	return clientset.NewForConfig(config)
}

// RandomSuffix provides a random sequence to append to pods,services,rcs.
func RandomSuffix() string {
	return strconv.Itoa(rand.Intn(10000))
}
