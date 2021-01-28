package util

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

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
func CopyFile(dstName, srcName string) (written int64, err error) {
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
