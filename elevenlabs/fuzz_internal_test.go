package elevenlabs

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func fixtureLinesInternal(tb testing.TB, name string) []string {
	tb.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		tb.Fatalf("read fixture: %v", err)
	}
	return strings.Split(strings.TrimSpace(string(raw)), "\n")
}

func FuzzDecodeSpeechMessage(f *testing.F) {
	for _, line := range fixtureLinesInternal(f, "tts_stream_messages.jsonl") {
		f.Add([]byte(line))
	}
	for _, seed := range []string{``, `{}`, `null`, `{"audio":"!!"}`, `{"audio":5}`, `{"error":"x"}`, `{"message":"m"}`, `{"isFinal":true}`, `[`} {
		f.Add([]byte(seed))
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		audio, done, err := decodeSpeechMessage(data)
		if err != nil && (audio != nil || done) {
			t.Fatalf("decodeSpeechMessage returned %q, %v together with error %v", audio, done, err)
		}
	})
}

func FuzzDecodeLiveMessage(f *testing.F) {
	for _, line := range fixtureLinesInternal(f, "stt_live_messages.jsonl") {
		f.Add([]byte(line), true)
		f.Add([]byte(line), false)
	}
	for _, seed := range []string{``, `{}`, `null`, `{"message_type":5}`, `{"message_type":"committed_transcript_with_timestamps","words":[{"start":1e308,"end":-1e308}]}`, `{"message_type":"x","error":"y"}`} {
		f.Add([]byte(seed), true)
	}
	f.Fuzz(func(t *testing.T, data []byte, timestamps bool) {
		seg, committed, err := decodeLiveMessage(data, timestamps)
		if err != nil && (seg != nil || committed) {
			t.Fatalf("decodeLiveMessage returned %v, %v together with error %v", seg, committed, err)
		}
	})
}

func FuzzParseError(f *testing.F) {
	for _, name := range []string{"tts_error_invalid_key.json", "tts_error_validation.json", "tts_error_quota.json"} {
		for _, line := range fixtureLinesInternal(f, name) {
			f.Add([]byte(line))
		}
	}
	for _, seed := range []string{``, `{}`, `{"detail":null}`, `{"detail":5}`, `{"detail":[]}`, `{"detail":[1]}`, `{"detail":"x"}`} {
		f.Add([]byte(seed))
	}
	f.Fuzz(func(_ *testing.T, data []byte) {
		parseError(data)
	})
}

func FuzzSegmentsOf(f *testing.F) {
	f.Add(`{"words":[{"text":"Hi.","type":"word","start":0,"end":1,"speaker_id":"a"},{"text":" ","type":"spacing"},{"text":"x","type":"audio_event"}]}`)
	f.Add(`{"words":[{"text":"a"},{"text":"b","speaker_id":"s"}]}`)
	f.Add(`{"words":[{"text":"","type":"word","start":1e308}]}`)
	f.Fuzz(func(t *testing.T, raw string) {
		var r scribeResponse
		if err := unmarshal(raw, &r); err != nil {
			return
		}
		for _, s := range segmentsOf(r.Words, "channel_0") {
			if strings.TrimSpace(s.Text) == "" {
				t.Fatalf("segmentsOf produced an empty segment %+v", s)
			}
		}
	})
}

func unmarshal(raw string, v any) error { return json.Unmarshal([]byte(raw), v) }
