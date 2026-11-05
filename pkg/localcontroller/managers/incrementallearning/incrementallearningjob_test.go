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

package incrementallearning

import "testing"

// TestTriggerEvalTaskMissingTrainedModel ensures triggerEvalTask returns an
// error instead of panicking with a nil pointer dereference when no trained
// model can be found in the job conditions (e.g. the train task didn't
// report a completed output model).
func TestTriggerEvalTaskMissingTrainedModel(t *testing.T) {
	im := &Manager{}
	job := &Job{
		JobConfig: &JobConfig{
			UniqueIdentifier: "default/test-job",
			EvalModel:        &Model{Format: "pb"},
		},
	}

	if _, err := im.triggerEvalTask(job); err == nil {
		t.Fatal("expected an error when the trained model is missing from job conditions, got nil")
	}
}
