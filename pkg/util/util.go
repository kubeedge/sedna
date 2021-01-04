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
