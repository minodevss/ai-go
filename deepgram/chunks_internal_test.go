package deepgram

import (
	"slices"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestSpeechChunks(t *testing.T) {
	long := strings.Repeat("word ", 500)
	tests := []struct {
		name   string
		pieces []string
		want   []string
	}{
		{name: "sentences across pieces", pieces: []string{"Hello wor", "ld. How are", " you? Fine"}, want: []string{"Hello world.", "How are you?", "Fine"}},
		{name: "newline ends a chunk", pieces: []string{"one\ntwo"}, want: []string{"one", "two"}},
		{name: "decimal point is not a boundary", pieces: []string{"Pi is 3.14 today"}, want: []string{"Pi is 3.14 today"}},
		{name: "blank input", pieces: []string{"  ", ""}, want: nil},
		{name: "long text without boundary is cut at a space", pieces: []string{long}, want: []string{strings.TrimSpace(long[:1999]), strings.TrimSpace(long[1999:])}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := slices.Collect(speechChunks(slices.Values(tt.pieces)))
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("speechChunks mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
