package google

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func batchSeedLines(tb testing.TB) []string {
	tb.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "batch", "results.jsonl"))
	if err != nil {
		tb.Fatalf("read fixture: %v", err)
	}
	return strings.Split(strings.TrimSpace(string(raw)), "\n")
}

func FuzzDecodeBatchResultLine(f *testing.F) {
	for _, line := range batchSeedLines(f) {
		f.Add([]byte(line))
	}
	for _, seed := range []string{``, `{}`, `null`, `[`, `{"key":1}`, `{"key":"a","error":"boom"}`, `{"key":"a","error":null}`, `{"key":"a","response":null}`, `{"key":"a","response":{"candidates":[]}}`, `{"key":"a","error":{"code":"x"}}`} {
		f.Add([]byte(seed))
	}
	m := New(WithAPIKey("k")).Model("m")
	f.Fuzz(func(t *testing.T, data []byte) {
		var rl resultLine
		if err := json.Unmarshal(data, &rl); err != nil {
			return
		}
		entry := m.entryOf(rl)
		if (entry.Response == nil) == (entry.Err == nil) {
			t.Fatalf("entryOf(%q) = %+v, want exactly one of response and error", data, entry)
		}
	})
}

func FuzzDecodeBatchOperation(f *testing.F) {
	for _, seed := range []string{
		`{"name":"batches/1","metadata":{"state":"JOB_STATE_RUNNING"}}`,
		`{"name":"batches/1","done":true,"response":{"inlinedResponses":{"inlinedResponses":[{"metadata":{"key":"a"},"error":{"code":3,"message":"x"}}]}}}`,
		`{"state":"BATCH_STATE_SUCCEEDED","dest":{"fileName":"files/x"}}`,
		``, `{}`, `[`, `{"metadata":null,"response":null}`,
	} {
		f.Add([]byte(seed))
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		var op batchOperation
		if err := json.Unmarshal(data, &op); err != nil {
			return
		}
		if state := batchState(op.view().State, op.Done); state == "" {
			t.Fatal("batchState returned an empty state")
		}
		op.output()
	})
}

func FuzzResultLines(f *testing.F) {
	for _, seed := range []string{"", "\n", "{}\n{}", "a\n\nb\n", strings.Repeat("x", 1<<17)} {
		f.Add([]byte(seed))
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		var got []byte
		for line, err := range resultLines(bytes.NewReader(data)) {
			if err != nil {
				t.Fatalf("resultLines() error = %v on an in-memory reader", err)
			}
			if len(line) == 0 {
				t.Fatal("resultLines() yielded an empty line")
			}
			got = append(got, line...)
		}
		want := strings.Join(strings.Fields(string(data)), "")
		if len(strings.Fields(string(got))) > 0 && len(strings.Join(strings.Fields(string(got)), "")) > len(want) {
			t.Fatalf("resultLines() yielded more bytes than the input held")
		}
	})
}
