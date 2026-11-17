package anthropic_test

import (
	"os"
	"path/filepath"
	"testing"
)

func fixture(tb testing.TB, name string) string {
	tb.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		tb.Fatalf("read fixture: %v", err)
	}
	return string(raw)
}
