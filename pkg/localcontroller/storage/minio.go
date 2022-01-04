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
	"context"
	"fmt"
	"net/url"
	"strings"
	"time"

	"k8s.io/klog/v2"

	"github.com/kubeedge/sedna/pkg/localcontroller/util"
	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
)

// MinioClient defines a minio client
type MinioClient struct {
	Client *minio.Client
}

// MaxTimeOut is max deadline time of client working
const MaxTimeOut = 100 * time.Second

// createMinioClient creates client
func createMinioClient(endpoint string, useHTTPS string, accessKeyID string, secretAccessKey string) (*MinioClient, error) {
	token := ""
	useSSL := true
	if useHTTPS == "0" {
		useSSL = false
	}

	client, err := minio.New(endpoint, &minio.Options{
		Creds:  credentials.NewStaticV4(accessKeyID, secretAccessKey, token),
		Secure: useSSL,
	})
	if err != nil {
		return nil, fmt.Errorf("initialize minio client failed, endpoint = %+v, error: %+v", endpoint, err)
	}

	c := MinioClient{
		Client: client,
	}

	return &c, nil
}

// uploadFile uploads file from local host to storage service
func (mc *MinioClient) uploadFile(localPath string, objectURL string) error {
	bucket, absPath, err := mc.parseURL(objectURL)
	if err != nil {
		return err
	}

	if !util.IsExists(localPath) {
		return fmt.Errorf("file(%s) in the local host is not exists", localPath)
	}

	ctx, cancel := context.WithTimeout(context.Background(), MaxTimeOut)
	defer cancel()

	if _, err = mc.Client.FPutObject(ctx, bucket, absPath, localPath, minio.PutObjectOptions{}); err != nil {
		return fmt.Errorf("upload file from file path(%s) to file url(%s) failed, error: %+v", localPath, objectURL, err)
	}

	return nil
}

// copyFile copies file from the bucket to another bucket in storage service
func (mc *MinioClient) copyFile(srcURL string, objectURL string) error {
	srcBucket, srcAbsPath, err := mc.parseURL(srcURL)
	if err != nil {
		return err
	}
	srcOptions := minio.CopySrcOptions{
		Bucket: srcBucket,
		Object: srcAbsPath,
	}

	objectBucket, objectAbsPath, err := mc.parseURL(objectURL)
	if err != nil {
		return err
	}
	objectOptions := minio.CopyDestOptions{
		Bucket: objectBucket,
		Object: objectAbsPath,
	}

	ctx, cancel := context.WithTimeout(context.Background(), MaxTimeOut)
	defer cancel()
	if _, err := mc.Client.CopyObject(ctx, objectOptions, srcOptions); err != nil {
		return fmt.Errorf("copy file from file url(%s) to file url(%s) failed, error: %+v", srcURL, objectURL, err)
	}

	return nil
}

// DownloadFile downloads file from storage service to the local host
func (mc *MinioClient) downloadFile(objectURL string, localPath string) error {
	bucket, absPath, err := mc.parseURL(objectURL)
	if err != nil {
		return err
	}

	ctx, cancel := context.WithTimeout(context.Background(), MaxTimeOut)
	defer cancel()

	if err = mc.Client.FGetObject(ctx, bucket, absPath, localPath, minio.GetObjectOptions{}); err != nil {
		return fmt.Errorf("download file from file url(%s) to file path(%s) failed, error: %+v", objectURL, localPath, err)
	}

	return nil
}

// parseURL parses url
func (mc *MinioClient) parseURL(URL string) (string, string, error) {
	u, err := url.Parse(URL)
	if err != nil {
		return "", "", fmt.Errorf("invalid url(%s)", URL)
	}

	scheme := u.Scheme
	switch scheme {
	case S3Prefix:
		return u.Host, strings.TrimPrefix(u.Path, "/"), nil
	default:
		klog.Errorf("invalid scheme(%s)", scheme)
	}

	return "", "", fmt.Errorf("invalid url(%s)", URL)
}

// deleteFile deletes file
func (mc *MinioClient) deleteFile(objectURL string) error {
	bucket, absPath, err := mc.parseURL(objectURL)
	if err != nil {
		return err
	}

	ctx, cancel := context.WithTimeout(context.Background(), MaxTimeOut)
	defer cancel()

	if err = mc.Client.RemoveObject(ctx, bucket, absPath, minio.RemoveObjectOptions{}); err != nil {
		return fmt.Errorf("delete file(url=%s) failed, error: %+v", objectURL, err)
	}

	return nil
}
