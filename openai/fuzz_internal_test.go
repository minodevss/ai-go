package openai

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func fixtureLinesInternal(tb testing.TB, name string) []string {
	tb.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "media", name))
	if err != nil {
		tb.Fatalf("read fixture: %v", err)
	}
	return strings.Split(strings.TrimSpace(string(raw)), "\n")
}

func FuzzDecodeServerEvent(f *testing.F) {
	for _, line := range fixtureLinesInternal(f, "realtime_server_events.jsonl") {
		f.Add([]byte(line))
	}
	for _, seed := range []string{``, `{}`, `null`, `{"type":"response.done"}`, `{"type":"response.done","response":null}`, `{"type":"response.output_audio.delta","delta":"!!"}`, `{"type":"error"}`, `{"type":"error","error":{"code":"response_cancel_not_active"}}`, `[`, `{"type":5}`} {
		f.Add([]byte(seed))
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		var w wireState
		events, err := decodeServerEvent(data, &w)
		if err != nil && len(events) != 0 {
			t.Fatalf("decodeServerEvent returned events %v together with error %v", events, err)
		}
	})
}

func FuzzDecodeTranscriptEvent(f *testing.F) {
	for _, seed := range []string{
		`{"type":"transcript.text.delta","delta":"Hello"}`,
		`{"type":"transcript.text.segment","id":"s","start":0,"end":1.5,"text":"Hi","speaker":"A"}`,
		`{"type":"transcript.text.done","text":"Hi"}`,
		`{"type":"error","error":{"code":"server_error","message":"x"}}`,
		`{"type":"error"}`, `{"type":"transcript.text.segment","start":1e308,"end":-1e308}`, ``, `{`,
	} {
		f.Add([]byte(seed))
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		saw := false
		seg, done, err := decodeTranscriptEvent(data, &saw)
		if err != nil && (seg != nil || done) {
			t.Fatalf("decodeTranscriptEvent returned %v, %v together with error %v", seg, done, err)
		}
	})
}

func FuzzDecodeLiveEvent(f *testing.F) {
	for _, seed := range []string{
		`{"type":"conversation.item.input_audio_transcription.delta","delta":"a"}`,
		`{"type":"conversation.item.input_audio_transcription.completed","transcript":"a"}`,
		`{"type":"conversation.item.input_audio_transcription.failed"}`,
		`{"type":"error","error":{"code":"x"}}`, ``, `{"type":null}`,
	} {
		f.Add([]byte(seed))
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		ev, err := decodeLiveEvent(data)
		if err != nil && (ev.seg != nil || ev.kind != liveNone) {
			t.Fatalf("decodeLiveEvent returned %v together with error %v", ev, err)
		}
	})
}

func FuzzBatchResultLine(f *testing.F) {
	for _, name := range []string{"batch_output.jsonl", "batch_error.jsonl"} {
		for _, line := range fixtureLinesInternal(f, name) {
			f.Add([]byte(line))
		}
	}
	for _, seed := range []string{``, `{}`, `{"custom_id":"a"}`, `{"custom_id":"a","response":{"status_code":200,"body":null}}`, `{"custom_id":"a","response":{"status_code":500,"body":"x"}}`, `{"response":{"status_code":200,"body":{"output":[{"type":"message","content":[{"type":"output_text"}]}]}}}`} {
		f.Add([]byte(seed))
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		dec := json.NewDecoder(bytes.NewReader(data))
		for {
			var line batchResultLine
			if err := dec.Decode(&line); err != nil {
				return
			}
			entry := entryOf(line)
			if (entry.Response == nil) == (entry.Err == nil) {
				t.Fatalf("entryOf(%+v) = %+v, want exactly one of Response and Err", line, entry)
			}
		}
	})
}
