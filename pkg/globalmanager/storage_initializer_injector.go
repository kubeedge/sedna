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
	"net/url"
	"path/filepath"
	"strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/klog/v2"
)

const (
	downloadInitalizerContainerName = "storage-initializer"
	downloadInitalizerImage         = "kubeedge/sedna-storage-initializer:v0.1.0"

	downloadInitalizerPrefix     = "/downloads"
	downloadInitalizerVolumeName = "sedna-storage-initializer"

	hostPathPrefix   = "/hostpath"
	urlsFieldSep     = ";"
	volumeNamePrefix = "sedna-"
)

var supportStorageInitializerURLSchemes = [...]string{
	// s3 compatible storage
	"s3",

	// http server, only for downloading
	"http", "https",
}

var supportURLSchemes = [...]string{
	// s3 compatbile storage
	"s3",

	// http server, only for downloading
	"http", "https",

	// hostpath of node, for compatibility only
	// "/opt/data/model.pb"
	"",

	// the local path of worker-container
	"file",
}

type workerMountMode string

const (
	workerMountReadOnly  workerMountMode = "readonly"
	workerMountWriteOnly workerMountMode = "writeonly"

// no read-write support for mount url/directory now
)

type MountURL struct {
	// URL is the url of dataset/model
	URL string

	// Mode indicates the url mode, default is workerMountReadOnly
	Mode workerMountMode

	// IsDir indicates that url is directory
	IsDir bool

	// if true, only mounts when url is hostpath
	CheckHostPath bool

	// the container path
	ContainerPath string

	// indicates the path this url will be mounted into container.
	// can be containerPath or its parent dir
	MountPath string

	// for host path, we just need to mount without downloading
	HostPath string

	// for storage initializer
	DownloadSrcURL string
	DownloadDstDir string

	// if Disable, then no mount
	Disable bool

	// parsed for the parent of url
	u *url.URL
}

func (m *MountURL) Parse() {
	u, _ := url.Parse(m.URL)

	m.u = u
	m.injectDownloadPath()
	m.injectHostPath()
}

func (m *MountURL) injectDownloadPath() {
	if m.Mode == workerMountWriteOnly {
		// no storage-initializer for write only
		// leave the write operation to worker
		return
	}

	for _, scheme := range supportStorageInitializerURLSchemes {
		if m.u.Scheme == scheme {
			m.MountPath = downloadInitalizerPrefix

			// here use u.Host + u.Path to avoid conflict
			m.ContainerPath = filepath.Join(m.MountPath, m.u.Host+m.u.Path)

			m.DownloadSrcURL = m.URL
			m.DownloadDstDir, _ = filepath.Split(m.ContainerPath)

			break
		}
	}
}

func (m *MountURL) injectHostPath() {
	// for compatibility, hostpath of a node is supported.
	// e.g. the url of a dataset: /datasets/d1/label.txt
	if m.u.Scheme != "" {
		if m.CheckHostPath {
			m.Disable = true
		}
		return
	}

	if m.IsDir {
		m.HostPath = m.URL
		m.MountPath = filepath.Join(hostPathPrefix, m.u.Path)
		m.ContainerPath = m.MountPath
	} else {
		// if file, here mount its directory
		m.HostPath, _ = filepath.Split(m.URL)
		m.ContainerPath = filepath.Join(hostPathPrefix, m.u.Path)
		m.MountPath, _ = filepath.Split(m.ContainerPath)
	}
}

func injectHostPathMount(pod *v1.Pod, workerParam *WorkerParam) {
	var volumes []v1.Volume
	var volumeMounts []v1.VolumeMount

	uniqVolumeName := make(map[string]bool)

	hostPathType := v1.HostPathDirectory

	for _, mount := range workerParam.mounts {
		for _, m := range mount.URLs {
			if m.HostPath != "" {
				volumeName := ConvertK8SValidName(m.HostPath)

				if volumeName == "" {
					klog.Warningf("failed to convert volume name from the url and skipped: %s", m.URL)
					continue
				}

				if _, ok := uniqVolumeName[volumeName]; !ok {
					volumes = append(volumes, v1.Volume{
						Name: volumeName,
						VolumeSource: v1.VolumeSource{
							HostPath: &v1.HostPathVolumeSource{
								Path: m.HostPath,
								Type: &hostPathType,
							},
						},
					})
					uniqVolumeName[volumeName] = true
				}

				vm := v1.VolumeMount{
					MountPath: m.MountPath,
					Name:      volumeName,
				}
				volumeMounts = append(volumeMounts, vm)
			}
		}
	}
	injectVolume(pod, volumes, volumeMounts)
}

func injectDownloadInitializer(pod *v1.Pod, workerParam *WorkerParam) {
	var volumes []v1.Volume
	var volumeMounts []v1.VolumeMount

	var downloadPairs []string
	for _, mount := range workerParam.mounts {
		for _, m := range mount.URLs {
			if m.DownloadSrcURL != "" && m.DownloadDstDir != "" {
				// srcURL dstDir
				// need to add srcURL first
				downloadPairs = append(downloadPairs, m.DownloadSrcURL, m.DownloadDstDir)
			}
		}
	}

	// no need to download
	if len(downloadPairs) == 0 {
		return
	}

	// use one empty directory
	storageVolume := v1.Volume{
		Name: downloadInitalizerVolumeName,
		VolumeSource: v1.VolumeSource{
			EmptyDir: &v1.EmptyDirVolumeSource{},
		},
	}

	storageVolumeMounts := v1.VolumeMount{
		Name:      storageVolume.Name,
		MountPath: downloadInitalizerPrefix,
		ReadOnly:  true,
	}
	volumes = append(volumes, storageVolume)
	volumeMounts = append(volumeMounts, storageVolumeMounts)

	initVolumeMounts := []v1.VolumeMount{
		{
			Name:      storageVolume.Name,
			MountPath: downloadInitalizerPrefix,
			ReadOnly:  false,
		},
	}

	initContainer := v1.Container{
		Name:            downloadInitalizerContainerName,
		Image:           downloadInitalizerImage,
		ImagePullPolicy: v1.PullIfNotPresent,
		Args:            downloadPairs,

		TerminationMessagePolicy: v1.TerminationMessageFallbackToLogsOnError,

		Resources: v1.ResourceRequirements{
			Limits: map[v1.ResourceName]resource.Quantity{
				// limit one cpu
				v1.ResourceCPU: resource.MustParse("1"),
				// limit 1Gi memory
				v1.ResourceMemory: resource.MustParse("1Gi"),
			},
		},
		VolumeMounts: initVolumeMounts,
	}

	pod.Spec.InitContainers = append(pod.Spec.InitContainers, initContainer)
	injectVolume(pod, volumes, volumeMounts)
}

// InjectStorageInitializer injects these storage related volumes and envs into pod in-place
func InjectStorageInitializer(pod *v1.Pod, workerParam *WorkerParam) {
	var mounts []WorkerMount
	// parse the mounts and environment key
	for _, mount := range workerParam.mounts {
		var envPaths []string

		if mount.URL != nil {
			mount.URLs = append(mount.URLs, *mount.URL)
		}

		var mountURLs []MountURL
		for _, m := range mount.URLs {
			m.Parse()
			if m.Disable {
				continue
			}
			mountURLs = append(mountURLs, m)
			envPaths = append(envPaths, m.ContainerPath)
		}

		if len(mountURLs) > 0 {
			mount.URLs = mountURLs
			mounts = append(mounts, mount)
		}
		if mount.EnvName != "" {
			workerParam.env[mount.EnvName] = strings.Join(
				envPaths, urlsFieldSep,
			)
		}
	}

	workerParam.mounts = mounts

	injectHostPathMount(pod, workerParam)
	injectDownloadInitializer(pod, workerParam)
}

func injectVolume(pod *v1.Pod, volumes []v1.Volume, volumeMounts []v1.VolumeMount) {
	if len(volumes) > 0 {
		pod.Spec.Volumes = append(pod.Spec.Volumes, volumes...)
	}

	if len(volumeMounts) > 0 {
		for idx := range pod.Spec.Containers {
			// inject every containers
			pod.Spec.Containers[idx].VolumeMounts = append(
				pod.Spec.Containers[idx].VolumeMounts, volumeMounts...,
			)
		}
	}
}
