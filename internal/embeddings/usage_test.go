package embeddings_test

import (
	"encoding/json"
	"testing"

	"github.com/minodevss/ai-go/internal/embeddings"
)

func TestTotalTokensUsage(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want int
	}{
		{name: "total", raw: `{"total_tokens":12}`, want: 12},
		{name: "missing", raw: `{}`, want: 0},
		{name: "invalid", raw: `[`, want: 0},
		{name: "empty", raw: ``, want: 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := embeddings.TotalTokensUsage(json.RawMessage(tt.raw))
			if got.Input != tt.want || string(got.Raw) != tt.raw {
				t.Fatalf("TotalTokensUsage(%q) = %+v, want input %d with raw kept", tt.raw, got, tt.want)
			}
		})
	}
}
