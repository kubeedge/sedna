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
