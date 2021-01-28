package v1alpha1

// Metric describes the data that a resource model metric should have
type Metric struct {
	Key   string `json:"key"`
	Value string `json:"value"`
}

// CommonWorkerSpec is a description of a worker both for edge and cloud
type CommonWorkerSpec struct {
	ScriptDir        string     `json:"scriptDir"`
	ScriptBootFile   string     `json:"scriptBootFile"`
	FrameworkType    string     `json:"frameworkType"`
	FrameworkVersion string     `json:"frameworkVersion"`
	Parameters       []ParaSpec `json:"parameters"`
}

// ParaSpec is a description of a parameter
type ParaSpec struct {
	Key   string `json:"key"`
	Value string `json:"value"`
}
