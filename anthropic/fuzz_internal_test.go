package anthropic

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func FuzzBatchResultLine(f *testing.F) {
	raw, err := os.ReadFile(filepath.Join("testdata", "batch", "results.jsonl"))
	if err != nil {
		f.Fatalf("read fixture: %v", err)
	}
	for line := range bytes.SplitSeq(bytes.TrimSpace(raw), []byte("\n")) {
		f.Add(line)
	}
	for _, seed := range []string{``, `{}`, `null`, `{"custom_id":"a","result":{"type":"succeeded"}}`, `{"custom_id":"a","result":{"type":"succeeded","message":{"content":[{"type":"tool_use"}],"stop_reason":"tool_use"}}}`, `{"result":{"type":"errored"}}`, `{"result":{"type":5}}`} {
		f.Add([]byte(seed))
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		var line batchResultLine
		if json.Unmarshal(data, &line) != nil {
			return
		}
		entry := entryOf(line)
		if (entry.Response == nil) == (entry.Err == nil) {
			t.Fatalf("entryOf() = %+v, want exactly one of Response and Err", entry)
		}
	})
}

func FuzzCachedBlock(f *testing.F) {
	for _, seed := range []string{``, `{}`, `null`, `[]`, `{"type":"text"}`, `{"type":"thinking","signature":"s"}`, `{"type":5}`, `{"type":"tool_use","cache_control":null}`} {
		f.Add([]byte(seed))
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		out := cachedBlock(data, &CacheControl{})
		if json.Valid(data) && !json.Valid(out) {
			t.Fatalf("cachedBlock(%q) = %q, want valid JSON", data, out)
		}
	})
}
