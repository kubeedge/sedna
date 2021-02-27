/*
Copyright 2021 The KubeEdge Authors.
Copyright 2016 The Kubernetes Authors.

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
	"flag"
	"os"

	"github.com/onsi/ginkgo/config"
	"k8s.io/client-go/tools/clientcmd"
)

type testContext struct {
	Master                   string
	KubeConfig               string
	DeleteNamespace          bool
	DeleteNamespaceOnFailure bool
}

var TestContext testContext

func RegisterFlags(flags *flag.FlagSet) {
	// Turn on verbose by default to get spec names
	config.DefaultReporterConfig.Verbose = true

	// Turn on EmitSpecProgress to get spec progress (especially on interrupt)
	config.GinkgoConfig.EmitSpecProgress = true

	// Randomize specs as well as suites
	config.GinkgoConfig.RandomizeAllSpecs = true

	flags.StringVar(&TestContext.Master, "master", os.Getenv("MASTER"), "The master, or apiserver, to connect to. Will default to MASTER environment if this argument and --kubeconfig are not set.")

	defaultKubeConfigPath := os.Getenv(clientcmd.RecommendedConfigPathEnvVar)
	if defaultKubeConfigPath == "" {
		defaultKubeConfigPath = clientcmd.RecommendedHomeFile
	}

	flags.StringVar(&TestContext.KubeConfig, clientcmd.RecommendedConfigPathFlag, defaultKubeConfigPath, "Path to kubeconfig containing embedded authinfo.")

	flags.BoolVar(&TestContext.DeleteNamespace, "delete-namespace", true, "If true tests will delete namespace after completion. It is only designed to make debugging easier, DO NOT turn it off by default.")
	flags.BoolVar(&TestContext.DeleteNamespaceOnFailure, "delete-namespace-on-failure", true, "If true, framework will delete test namespace on failure. Used only during test debugging.")
}
