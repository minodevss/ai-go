package assemblyai

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func FuzzDecodeServerMessage(f *testing.F) {
	raw, err := os.ReadFile(filepath.Join("testdata", "streaming_turns.jsonl"))
	if err != nil {
		f.Fatalf("read fixture: %v", err)
	}
	for line := range strings.SplitSeq(strings.TrimSpace(string(raw)), "\n") {
		f.Add([]byte(line))
	}
	for _, seed := range []string{``, `{}`, `null`, `[`, `{"type":5}`, `{"type":"Turn","words":null}`, `{"type":"Turn","transcript":"x","words":[{"start":9223372036854775807,"end":-9223372036854775808}]}`, `{"type":"Error"}`, `{"type":"Error","error_code":"3006","error":"x"}`, `{"error":"x"}`} {
		f.Add([]byte(seed))
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		seg, done, err := decodeServerMessage(data)
		if err != nil && (seg != nil || done) {
			t.Fatalf("decodeServerMessage returned %v, %v together with error %v", seg, done, err)
		}
	})
}

func FuzzFormatOf(f *testing.F) {
	for _, seed := range []string{``, `audio/pcm`, `audio/pcm;rate=16000`, `audio/basic;rate=8000`, `audio/pcm;rate=abc`, `audio/pcm;rate=1`, `;`, `audio/mpeg`} {
		f.Add(seed)
	}
	f.Fuzz(func(t *testing.T, mediaType string) {
		got, err := formatOf(mediaType)
		if err == nil && (got.chunkBytes <= 0 || got.minBytes <= 0 || got.minBytes > got.chunkBytes) {
			t.Fatalf("formatOf(%q) = %+v, want positive chunk sizes", mediaType, got)
		}
	})
}
