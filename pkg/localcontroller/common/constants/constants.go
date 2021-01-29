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

package constants

const (
	// ServerRootPath is the root path of server
	ServerRootPath = "sedna"

	// DataBaseURL is url of database
	DataBaseURL = "/var/lib/sedna/database.db"

	// WSScheme is the scheme of websocket
	WSScheme = "ws"

	// WSHeaderNodeName is the name of header of websocket
	WSHeaderNodeName = "Node-Name"

	// GMAddressENV is the env name of address of GM
	GMAddressENV = "GM_ADDRESS"

	// NodeNameENV is the env of name of node of LC
	NodeNameENV = "NODENAME"

	// HostNameENV is the env of name of host
	HostNameENV = "HOSTNAME"

	// RootFSMountDirENV is the env of dir of mount
	RootFSMountDirENV = "ROOTFS_MOUNT_DIR"

	// BindPortENV is the env of binding port
	BindPortENV = "BIND_PORT"
)
