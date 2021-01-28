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
	"os"
	"strings"
)

// FileIsExist check file is exist
func FileIsExist(path string) bool {
	_, err := os.Stat(path)
	if err == nil {
		return true
	}
	return os.IsExist(err)
}

func SpliceErrors(errors []error) string {
	if len(errors) == 0 {
		return ""
	}
	var stb strings.Builder
	stb.WriteString("[\n")
	for _, err := range errors {
		stb.WriteString(fmt.Sprintf("  %s\n", err.Error()))
	}
	stb.WriteString("]\n")
	return stb.String()
}
