package deepgram

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unicode/utf8"
)

func FuzzDecodeLiveMessage(f *testing.F) {
	raw, err := os.ReadFile(filepath.Join("testdata", "live_messages.jsonl"))
	if err != nil {
		f.Fatalf("read fixture: %v", err)
	}
	for line := range strings.SplitSeq(strings.TrimSpace(string(raw)), "\n") {
		f.Add([]byte(line))
	}
	for _, seed := range []string{``, `{}`, `null`, `[`, `{"type":"Results"}`, `{"type":"Results","channel":[0,1]}`, `{"type":"Results","channel":{"alternatives":[]}}`, `{"type":"Results","start":1e308,"duration":1e308,"channel":{"alternatives":[{"transcript":"a"}]}}`, `{"type":"Error"}`, `{"type":5}`} {
		f.Add([]byte(seed))
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		segs, err := decodeLiveMessage(data)
		if err != nil && len(segs) != 0 {
			t.Fatalf("decodeLiveMessage returned %v together with error %v", segs, err)
		}
	})
}

func FuzzListenResponse(f *testing.F) {
	for _, name := range []string{"listen_response.json", "listen_paragraphs.json"} {
		raw, err := os.ReadFile(filepath.Join("testdata", name))
		if err != nil {
			f.Fatalf("read fixture: %v", err)
		}
		f.Add(raw)
	}
	for _, seed := range []string{``, `{}`, `{"results":null}`, `{"results":{"channels":[{"alternatives":[]}]}}`, `{"results":{"utterances":[{"start":-1e308}]}}`} {
		f.Add([]byte(seed))
	}
	f.Fuzz(func(_ *testing.T, data []byte) {
		var body listenResponse
		if json.Unmarshal(data, &body) != nil {
			return
		}
		responseOf(body, nil)
	})
}

func FuzzSpeechChunks(f *testing.F) {
	for _, seed := range []string{"", "Hello. World.", "no boundary at all", strings.Repeat("word ", 1200), strings.Repeat("あ", 2500), "One.\nTwo! Three? ", "\xff\xfe"} {
		f.Add(seed, 7)
	}
	f.Fuzz(func(t *testing.T, text string, size int) {
		size = max(size, 1)
		pieces := func(yield func(string) bool) {
			for i := 0; i < len(text); i += size {
				if !yield(text[i:min(i+size, len(text))]) {
					return
				}
			}
		}
		total := 0
		for chunk := range speechChunks(pieces) {
			if n := utf8.RuneCountInString(chunk); n > maxSpeechChars || strings.TrimSpace(chunk) == "" {
				t.Fatalf("chunk of %d runes %q is empty or over the limit", n, chunk)
			}
			total += len(strings.Fields(chunk))
		}
		if utf8.ValidString(text) && total > len(strings.Fields(text))+len(text)/maxSpeechChars+1 {
			t.Fatalf("chunks hold %d words, input has %d", total, len(strings.Fields(text)))
		}
	})
}
