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

package storage

import (
	"encoding/json"
	"fmt"
	"net/url"
	"path"
	"path/filepath"

	"k8s.io/klog/v2"

	"github.com/kubeedge/sedna/pkg/localcontroller/util"
)

const (
	// S3Prefix defines that prefix of url is s3
	S3Prefix = "s3"
	// LocalPrefix defines that prefix of url is local host
	LocalPrefix = ""
	// S3EndPoint is s3 endpoint of the storage service
	S3Endpoint = "s3-endpoint"
	// S3UseHTTPS determines whether to use HTTPS protocol
	S3UseHTTPS = "s3-usehttps"
	// AccessKeyId is access key id of the storage service
	AccessKeyID = "ACCESS_KEY_ID"
	// SecretAccessKey is secret access key of the storage service
	SecretAccessKey = "SECRET_ACCESS_KEY"
)

type Storage struct {
	MinioClient *MinioClient
}

// Download downloads the file to the local host
func (s *Storage) Download(objectURL string, localPath string) (string, error) {
	prefix, err := CheckURL(objectURL)
	if err != nil {
		return "", err
	}

	switch prefix {
	case LocalPrefix:
		return s.localCopy(objectURL, localPath)
	case S3Prefix:
		return s.downloadS3(objectURL, localPath)
	default:
		return "", fmt.Errorf("invalid url(%s)", objectURL)
	}
}

// downloadLocal copies the local file to another in local host
func (s *Storage) localCopy(objectURL string, localPath string) (string, error) {
	if !util.IsExists(objectURL) {
		return "", fmt.Errorf("url(%s) is not exists", objectURL)
	}

	if localPath == "" {
		return objectURL, nil
	}

	dir := path.Dir(localPath)
	if !util.IsDir(dir) {
		util.CreateFolder(dir)
	}

	util.CopyFile(objectURL, localPath)

	return localPath, nil
}

// downloadS3 downloads the file from url of s3 to the local host
func (s *Storage) downloadS3(objectURL string, localPath string) (string, error) {
	if localPath == "" {
		temporaryDir, err := util.CreateTemporaryDir()
		if err != nil {
			return "", err
		}

		localPath = path.Join(temporaryDir, filepath.Base(objectURL))
	}

	dir := path.Dir(localPath)
	if !util.IsDir(dir) {
		util.CreateFolder(dir)
	}

	if err := s.MinioClient.downloadFile(objectURL, localPath); err != nil {
		return "", err
	}

	return localPath, nil
}

// checkMapKey checks whether key exists in the dict
func checkMapKey(m map[string]string, key string) (string, error) {
	v, ok := m[key]
	if !ok {
		return "", fmt.Errorf("%s is empty", key)
	}
	return v, nil
}

// SetCredential sets credential of the storage service
func (s *Storage) SetCredential(credential string) error {
	c := credential
	m := make(map[string]string)
	if err := json.Unmarshal([]byte(c), &m); err != nil {
		return err
	}

	endpoint, err := checkMapKey(m, S3Endpoint)
	if err != nil {
		return err
	}

	useHTTPS, err := checkMapKey(m, S3UseHTTPS)
	if err != nil {
		useHTTPS = "1"
	}

	ak, err := checkMapKey(m, AccessKeyID)
	if err != nil {
		return err
	}

	sk, err := checkMapKey(m, SecretAccessKey)
	if err != nil {
		return err
	}

	mc, err := createMinioClient(endpoint, useHTTPS, ak, sk)
	if err != nil {
		return err
	}

	s.MinioClient = mc

	return nil
}

// Upload uploads the file in local host to another(e.g., "s3")
func (s *Storage) Upload(localPath string, objectURL string) error {
	prefix, err := CheckURL(objectURL)
	if err != nil {
		return err
	}

	switch prefix {
	case S3Prefix:
		return s.uploadS3(localPath, objectURL)
	default:
		return fmt.Errorf("invalid url(%s)", objectURL)
	}
}

// uploadS3 uploads the file in local host to the url of s3
func (s *Storage) uploadS3(srcURL string, objectURL string) error {
	prefix, err := CheckURL(srcURL)
	if err != nil {
		return err
	}
	switch prefix {
	case LocalPrefix:
		if err := s.MinioClient.uploadFile(srcURL, objectURL); err != nil {
			return err
		}
	case S3Prefix:
		if err := s.MinioClient.copyFile(srcURL, objectURL); err != nil {
			return err
		}
	}
	return nil
}

// CheckURL checks prefix of the url
func CheckURL(objectURL string) (string, error) {
	if objectURL == "" {
		return "", fmt.Errorf("empty url")
	}

	u, err := url.Parse(objectURL)
	if err != nil {
		klog.Errorf("invalid url(%s), error: %v", objectURL, err)
		return "", fmt.Errorf("invalid url(%s)", objectURL)
	}

	l := []string{LocalPrefix, S3Prefix}

	for _, v := range l {
		if u.Scheme == v {
			return u.Scheme, nil
		}
	}

	return "", fmt.Errorf("unvalid url(%s), not support prefix(%s)", objectURL, u.Scheme)
}
