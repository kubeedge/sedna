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

package util

import (
	"fmt"
	"io"
	"math/rand"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"k8s.io/klog/v2"
)

// IsExists checks whether the file or folder exists
func IsExists(path string) bool {
	_, err := os.Stat(path)
	if err != nil {
		return os.IsExist(err)
	}
	return true
}

// IsFile checks whether the specified path is a file
func IsFile(path string) bool {
	return !IsDir(path)
}

// IsDir checks whether the given path is a folder
func IsDir(path string) bool {
	s, err := os.Stat(path)
	if err != nil {
		return false
	}
	return s.IsDir()
}

// CopyFile copies a file to other
func CopyFile(srcName, dstName string) (written int64, err error) {
	src, err := os.Open(srcName)
	if err != nil {
		klog.Errorf("open file %s failed: %v", srcName, err)
		return -1, err
	}
	defer func() {
		_ = src.Close()
	}()

	dst, err := os.OpenFile(dstName, os.O_WRONLY|os.O_CREATE, 0644)
	if err != nil {
		return -1, err
	}

	defer func() {
		_ = dst.Close()
	}()

	return io.Copy(dst, src)
}

// CreateFolder creates a fold
func CreateFolder(path string) error {
	s, err := os.Stat(path)
	if os.IsNotExist(err) {
		if err := os.MkdirAll(path, os.ModePerm); err != nil {
			return err
		}
		return nil
	}

	if !s.IsDir() {
		if err := os.MkdirAll(path, os.ModePerm); err != nil {
			return err
		}
	}

	return nil
}

// TrimPrefixPath gets path without the provided leading prefix path. If path doesn't start with prefix, path is returned.
func TrimPrefixPath(prefix string, path string) string {
	return strings.TrimPrefix(path, prefix)
}

// AddPrefixPath gets path that adds the provided the prefix.
func AddPrefixPath(prefix string, path string) string {
	return filepath.Join(prefix, path)
}

// GetUniqueIdentifier get unique identifier
func GetUniqueIdentifier(namespace string, name string, kind string) string {
	return fmt.Sprintf("%s/%s/%s", namespace, kind, name)
}

// CreateTemporaryDir creates a temporary dir
func CreateTemporaryDir() (string, error) {
	var src = rand.NewSource(time.Now().UnixNano())
	dir := path.Join("/tmp/", strconv.FormatInt(src.Int63(), 10), "/")
	err := CreateFolder(dir)
	return dir, err
}

// ParsingDatasetIndex parses index file of dataset and adds the prefix to abs url of sample
// first line is the prefix, the next lines are abs url of sample
func ParsingDatasetIndex(samples []string, prefix string) []string {
	var l []string
	l = append(l, prefix)
	for _, v := range samples {
		tmp := strings.Split(v, " ")
		for _, data := range tmp {
			if path.Ext(data) != "" {
				l = append(l, data)
			}
		}
	}

	return l
}
