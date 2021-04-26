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
	"encoding/json"
	"fmt"

	v1 "k8s.io/api/core/v1"
)

const (
	S3EndpointKey = "s3-endpoint"
	S3UseHTTPSKey = "s3-usehttps"
	// env name
	S3EndpointURLEnv = "S3_ENDPOINT_URL"

	AccessKeyID = "ACCESS_KEY_ID"
	// env name
	AccessKeyIDEnv = "ACCESS_KEY_ID"

	SecretAccessKey = "SECRET_ACCESS_KEY"
	// env name
	SecretAccessKeyEnv = "SECRET_ACCESS_KEY"

	SecretAnnotationKey = "sedna.io/credential"
)

func buildS3SecretEnvs(secret *v1.Secret) (envs []v1.EnvVar, err error) {
	if secret == nil {
		return
	}

	var s3Endpoint string
	if s3Endpoint = secret.Annotations[S3EndpointKey]; s3Endpoint == "" {
		err = fmt.Errorf("empty endpoint in secret %s", secret.Name)
		return
	}

	var s3EndpointURL string
	useHTTPS := true
	if httpsConf, ok := secret.Annotations[S3UseHTTPSKey]; ok && httpsConf == "0" {
		useHTTPS = false
	}

	if useHTTPS {
		s3EndpointURL = "https://" + s3Endpoint
	} else {
		s3EndpointURL = "http://" + s3Endpoint
	}

	envs = append(envs, v1.EnvVar{
		Name:  S3EndpointURLEnv,
		Value: s3EndpointURL,
	})

	// Better to use secretKeyRef or EnvFrom for this.
	// But now(2021/4/28) kubeedge does not support secretKeyRef for env value,
	// there is a open pr for this, https://github.com/kubeedge/kubeedge/pull/2230
	envs = append(envs,
		v1.EnvVar{
			Name:  AccessKeyIDEnv,
			Value: string(secret.Data[AccessKeyID]),
		},
	)
	envs = append(envs,
		v1.EnvVar{
			Name:  SecretAccessKeyEnv,
			Value: string(secret.Data[SecretAccessKey]),
		},
	)
	return
}

// MergeSecretEnvs merges two EnvVar list
func MergeSecretEnvs(nowE, newE []v1.EnvVar, overwrite bool) []v1.EnvVar {
	existEnvNames := make(map[string]int)
	for i, e := range nowE {
		existEnvNames[e.Name] = i
	}

	for _, e := range newE {
		if idx, exist := existEnvNames[e.Name]; !exist {
			nowE = append(nowE, e)
			existEnvNames[e.Name] = len(nowE) - 1
		} else {
			if overwrite {
				nowE[idx] = e
			}
		}
	}
	return nowE
}

func InjectSecretObj(obj CommonInterface, secret *v1.Secret) {
	if secret == nil {
		return
	}

	secretData := secret.GetAnnotations()

	for k, v := range secret.Data {
		// v already decoded
		secretData[k] = string(v)
	}

	b, _ := json.Marshal(secretData)
	ann := obj.GetAnnotations()
	if ann == nil {
		ann = make(map[string]string)
	}

	ann[SecretAnnotationKey] = string(b)

	obj.SetAnnotations(ann)
}
