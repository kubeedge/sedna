package options

// LocalControllerOptions defines options
type LocalControllerOptions struct {
	GMAddr            string
	NodeName          string
	BindPort          string
	VolumeMountPrefix string
}

// NewLocalControllerOptions create options object
func NewLocalControllerOptions() *LocalControllerOptions {
	return &LocalControllerOptions{}
}
