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

package apps

import (
	"context"
	"fmt"

	"github.com/onsi/ginkgo"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	sednav1alpha1 "github.com/kubeedge/sedna/pkg/apis/sedna/v1alpha1"
	sednaclient "github.com/kubeedge/sedna/pkg/client/clientset/versioned"

	"github.com/kubeedge/sedna/test/e2e/framework"
)

var _ = ginkgo.Describe("Dataset", func() {
	f := framework.NewDefaultFramework("dataset")

	// TODO(llhuii): find a better way to skip this hard code
	// the kubeedge default edge node name
	testNodeName := "edge-node"

	var sc sednaclient.Interface
	var ns string

	ginkgo.BeforeEach(func() {
		sc = f.SednaClientSet
		ns = f.Namespace.Name
	})

	ginkgo.AfterEach(func() {
		// Clean up
		datasets, err := sc.SednaV1alpha1().Datasets(ns).List(context.TODO(), metav1.ListOptions{})
		if datasets != nil && len(datasets.Items) > 0 {
			for _, ds := range datasets.Items {
				ginkgo.By(fmt.Sprintf("Deleting Dataset %s", ds.Name))
				err = deleteDateset(sc, ns, ds.Name)
				framework.ExpectNoError(err, "error deleting dataset")

			}
		}
	})

	ginkgo.Describe("Basic Dataset functionality", func() {
		ginkgo.It("Create a new dataset", func() {
			dataset := newDataset("test", "/e2e/d1.txt", testNodeName)
			_, err := createDataset(sc, ns, dataset)
			framework.ExpectNoError(err)
		})
	})
})

func createDataset(c sednaclient.Interface, ns string, d *sednav1alpha1.Dataset) (*sednav1alpha1.Dataset, error) {
	return c.SednaV1alpha1().Datasets(ns).Create(context.TODO(), d, metav1.CreateOptions{})
}

func getDataset(c sednaclient.Interface, ns, name string) (*sednav1alpha1.Dataset, error) {
	return c.SednaV1alpha1().Datasets(ns).Get(context.TODO(), name, metav1.GetOptions{})
}

func deleteDateset(c sednaclient.Interface, ns, name string) error {
	return c.SednaV1alpha1().Datasets(ns).Delete(context.TODO(), name, metav1.DeleteOptions{})
}

func newDataset(name, url, nodeName string) *sednav1alpha1.Dataset {
	return &sednav1alpha1.Dataset{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: sednav1alpha1.DatasetSpec{
			URL:      url,
			NodeName: nodeName,
			Format:   "txt",
		},
	}
}
