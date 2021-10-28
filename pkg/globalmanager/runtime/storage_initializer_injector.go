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
	"net/url"
	"path/filepath"
	"strings"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/klog/v2"
)

const (
	downloadInitalizerContainerName = "storage-initializer"
	downloadInitalizerImage         = "kubeedge/sedna-storage-initializer:v0.3.0"

	downloadInitalizerPrefix     = "/downloads"
	downloadInitalizerVolumeName = "sedna-storage-initializer"

	hostPathPrefixEnvKey = "DATA_PATH_PREFIX"
	hostPathPrefix       = "/home/data"
	urlsFieldSep         = ";"

	indirectURLMark    = "@"
	indirectURLMarkEnv = "INDIRECT_URL_MARK"

	defaultVolumeName = "sedna-default-volume-name"
)

var supportStorageInitializerURLSchemes = [...]string{
	// s3 compatible storage
	"s3",

	// http server, only for downloading
	"http", "https",
}

type MountURL struct {
	// URL is the url of dataset/model
	URL string

	// Indirect indicates the url is indirect, need to parse its content and download all,
	// and is used in dataset which has index url.
	//
	// when Indirect = true, URL could be in host path filesystem.
	// default: false
	Indirect bool

	// DownloadByInitializer indicates whether the url need to be download by initializer.
	DownloadByInitializer bool

	// IsDir indicates that url is directory
	IsDir bool

	// if true, only mounts when url is hostpath
	EnableIfHostPath bool

	// the container path
	ContainerPath string

	// indicates the path this url will be mounted into container.
	// can be ContainerPath or its parent dir
	MountPath string

	// for host path, we just need to mount without downloading
	HostPath string

	// for download
	DownloadSrcURL string
	DownloadDstDir string

	// if true, then no mount
	Disable bool

	// the relevant secret
	Secret     *v1.Secret
	SecretEnvs []v1.EnvVar

	// parsed for the parent of url
	u *url.URL
}

func (m *MountURL) Parse() {
	u, _ := url.Parse(m.URL)

	m.u = u
	m.parseDownloadPath()
	m.parseHostPath()
	m.parseSecret()
}

func (m *MountURL) parseDownloadPath() {
	if !m.DownloadByInitializer {
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

func (m *MountURL) parseHostPath() {
	// for compatibility, hostpath of a node is supported.
	// e.g. the url of a dataset: /datasets/d1/label.txt
	if m.u.Scheme != "" {
		if m.EnableIfHostPath {
			// not hostpath, so disable it
			m.Disable = true
		}
		return
	}

	if m.IsDir {
		m.HostPath = m.URL
		m.MountPath = filepath.Join(hostPathPrefix, m.u.Path)
		m.ContainerPath = m.MountPath
	} else {
		// if file, here mount its parent directory
		m.HostPath, _ = filepath.Split(m.URL)
		m.ContainerPath = filepath.Join(hostPathPrefix, m.u.Path)
		m.MountPath, _ = filepath.Split(m.ContainerPath)
		if m.Indirect {
			// we need to download it
			// TODO: mv these to download-related section
			m.DownloadSrcURL = m.ContainerPath
			m.ContainerPath = filepath.Join(downloadInitalizerPrefix, m.u.Host+m.u.Path)
			m.DownloadDstDir, _ = filepath.Split(m.ContainerPath)
		}
	}
}

func (m *MountURL) parseSecret() {
	if m.Secret == nil {
		return
	}

	if strings.ToLower(m.u.Scheme) == "s3" || m.Indirect {
		SecretEnvs, err := buildS3SecretEnvs(m.Secret)
		if err == nil {
			m.SecretEnvs = SecretEnvs
		}
	}
}

func injectHostPathMount(pod *v1.Pod, workerParam *WorkerParam) {
	volumes, volumeMounts, initContainerVolumeMounts := PrepareHostPath(workerParam)

	injectVolume(pod, volumes, volumeMounts)

	if len(volumeMounts) > 0 {
		hostPathEnvs := []v1.EnvVar{
			{
				Name:  hostPathPrefixEnvKey,
				Value: hostPathPrefix,
			},
		}
		injectEnvs(pod, hostPathEnvs)
	}

	if len(initContainerVolumeMounts) > 0 {
		initIdx := len(pod.Spec.InitContainers) - 1
		pod.Spec.InitContainers[initIdx].VolumeMounts = append(
			pod.Spec.InitContainers[initIdx].VolumeMounts,
			initContainerVolumeMounts...,
		)
	}
}

func injectWorkerSecrets(pod *v1.Pod, workerParam *WorkerParam) {
	var secretEnvs []v1.EnvVar
	for _, mount := range workerParam.Mounts {
		for _, m := range mount.URLs {
			if m.Disable || m.DownloadByInitializer {
				continue
			}
			if len(m.SecretEnvs) > 0 {
				secretEnvs = MergeSecretEnvs(secretEnvs, m.SecretEnvs, false)
			}
		}
	}
	injectEnvs(pod, secretEnvs)
}

func injectInitializerContainer(pod *v1.Pod, workerParam *WorkerParam) {
	volumes, volumeMounts, initContainer := PrepareInitContainer(workerParam)

	if (len(volumes) > 0) && (len(volumeMounts) > 0) && &initContainer != nil {
		pod.Spec.InitContainers = append(pod.Spec.InitContainers, *initContainer)
		injectVolume(pod, volumes, volumeMounts)
	}
}

/*
 Deployment Storage Hooks
*/

func injectHostPathMountDeployment(deployment *appsv1.Deployment, workerParam *WorkerParam) {
	volumes, volumeMounts, initContainerVolumeMounts := PrepareHostPath(workerParam)
	injectVolumeDeployment(deployment, volumes, volumeMounts)

	if len(volumeMounts) > 0 {
		hostPathEnvs := []v1.EnvVar{
			{
				Name:  hostPathPrefixEnvKey,
				Value: hostPathPrefix,
			},
		}
		injectEnvsDeployment(deployment, hostPathEnvs)
	}

	if len(initContainerVolumeMounts) > 0 {
		initIdx := len(deployment.Spec.Template.Spec.InitContainers) - 1
		deployment.Spec.Template.Spec.InitContainers[initIdx].VolumeMounts = append(
			deployment.Spec.Template.Spec.InitContainers[initIdx].VolumeMounts,
			initContainerVolumeMounts...,
		)
	}
}

func injectWorkerSecretsDeployment(deployment *appsv1.Deployment, workerParam *WorkerParam) {
	secretEnvs := PrepareSecret(workerParam)
	injectEnvsDeployment(deployment, secretEnvs)
}

func injectInitializerContainerDeployment(deployment *appsv1.Deployment, workerParam *WorkerParam) {
	volumes, volumeMounts, initContainer := PrepareInitContainer(workerParam)

	if (len(volumes) > 0) && (len(volumeMounts) > 0) && &initContainer != nil {
		deployment.Spec.Template.Spec.InitContainers = append(deployment.Spec.Template.Spec.InitContainers, *initContainer)
		injectVolumeDeployment(deployment, volumes, volumeMounts)
	}
}

// InjectStorageInitializer injects these storage related volumes and envs into deployment in-place
func InjectStorageInitializerDeployment(deployment *appsv1.Deployment, workerParam *WorkerParam) {
	PrepareStorage(workerParam)

	// need to call injectInitializerContainer before injectHostPathMount
	// since injectHostPathMount could inject volumeMount to init container
	injectInitializerContainerDeployment(deployment, workerParam)
	injectHostPathMountDeployment(deployment, workerParam)
	injectWorkerSecretsDeployment(deployment, workerParam)
}

func injectVolumeDeployment(deployment *appsv1.Deployment, volumes []v1.Volume, volumeMounts []v1.VolumeMount) {
	if len(volumes) > 0 {
		deployment.Spec.Template.Spec.Volumes = append(deployment.Spec.Template.Spec.Volumes, volumes...)
	}

	if len(volumeMounts) > 0 {
		for idx := range deployment.Spec.Template.Spec.Containers {
			// inject every containers
			deployment.Spec.Template.Spec.Containers[idx].VolumeMounts = append(
				deployment.Spec.Template.Spec.Containers[idx].VolumeMounts, volumeMounts...,
			)
		}
	}
}

func injectEnvsDeployment(deployment *appsv1.Deployment, envs []v1.EnvVar) {
	if len(envs) > 0 {
		for idx := range deployment.Spec.Template.Spec.Containers {
			// inject every containers
			deployment.Spec.Template.Spec.Containers[idx].Env = append(
				deployment.Spec.Template.Spec.Containers[idx].Env, envs...,
			)
		}
	}
}

// InjectStorageInitializer injects these storage related volumes and envs into pod in-place
func InjectStorageInitializer(pod *v1.Pod, workerParam *WorkerParam) {
	PrepareStorage(workerParam)

	// need to call injectInitializerContainer before injectHostPathMount
	// since injectHostPathMount could inject volumeMount to init container
	injectInitializerContainer(pod, workerParam)
	injectHostPathMount(pod, workerParam)
	injectWorkerSecrets(pod, workerParam)
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

func injectEnvs(pod *v1.Pod, envs []v1.EnvVar) {
	if len(envs) > 0 {
		for idx := range pod.Spec.Containers {
			// inject every containers
			pod.Spec.Containers[idx].Env = append(
				pod.Spec.Containers[idx].Env, envs...,
			)
		}
	}
}

func PrepareStorage(workerParam *WorkerParam) {
	var mounts []WorkerMount
	// parse the mounts and environment key
	for _, mount := range workerParam.Mounts {
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

			if m.ContainerPath != "" {
				envPaths = append(envPaths, m.ContainerPath)
			} else {
				// keep the original URL if no container path
				envPaths = append(envPaths, m.URL)
			}
		}

		if len(mountURLs) > 0 {
			mount.URLs = mountURLs
			mounts = append(mounts, mount)
		}

		if mount.EnvName != "" {
			workerParam.Env[mount.EnvName] = strings.Join(
				envPaths, urlsFieldSep,
			)
		}
	}

	workerParam.Mounts = mounts
}

func PrepareSecret(workerParam *WorkerParam) []v1.EnvVar {
	var secretEnvs []v1.EnvVar
	for _, mount := range workerParam.Mounts {
		for _, m := range mount.URLs {
			if m.Disable || m.DownloadByInitializer {
				continue
			}
			if len(m.SecretEnvs) > 0 {
				secretEnvs = MergeSecretEnvs(secretEnvs, m.SecretEnvs, false)
			}
		}
	}

	return secretEnvs
}

func PrepareHostPath(workerParam *WorkerParam) ([]v1.Volume, []v1.VolumeMount, []v1.VolumeMount) {
	var volumes []v1.Volume
	var volumeMounts []v1.VolumeMount
	var initContainerVolumeMounts []v1.VolumeMount

	uniqVolumeName := make(map[string]bool)

	hostPathType := v1.HostPathDirectory

	for _, mount := range workerParam.Mounts {
		for _, m := range mount.URLs {
			if m.HostPath == "" {
				continue
			}

			volumeName := ConvertK8SValidName(m.HostPath)

			if len(volumeName) == 0 {
				volumeName = defaultVolumeName
				klog.Warningf("failed to get name from url(%s), fallback to default name(%s)", m.URL, volumeName)
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
			if m.Indirect {
				initContainerVolumeMounts = append(initContainerVolumeMounts, vm)
			} else {
				volumeMounts = append(volumeMounts, vm)
			}
		}
	}

	return volumes, volumeMounts, initContainerVolumeMounts
}

func PrepareInitContainer(workerParam *WorkerParam) ([]v1.Volume, []v1.VolumeMount, *v1.Container) {
	var volumes []v1.Volume
	var volumeMounts []v1.VolumeMount

	var downloadPairs []string
	var secretEnvs []v1.EnvVar
	for _, mount := range workerParam.Mounts {
		for _, m := range mount.URLs {
			if m.Disable {
				continue
			}

			srcURL := m.DownloadSrcURL
			dstDir := m.DownloadDstDir
			if srcURL != "" && dstDir != "" {
				// need to add srcURL first: srcURL dstDir
				if m.Indirect {
					// here add indirectURLMark into dstDir which is controllable
					dstDir = indirectURLMark + dstDir
				}
				downloadPairs = append(downloadPairs, srcURL, dstDir)

				if len(m.SecretEnvs) > 0 {
					secretEnvs = MergeSecretEnvs(secretEnvs, m.SecretEnvs, false)
				}
			}
		}
	}

	// no need to download
	if len(downloadPairs) == 0 {
		return nil, nil, nil
	}

	envs := secretEnvs
	envs = append(envs, v1.EnvVar{
		Name:  indirectURLMarkEnv,
		Value: indirectURLMark,
	})

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
		Env:          envs,
	}

	return volumes, volumeMounts, &initContainer
}
